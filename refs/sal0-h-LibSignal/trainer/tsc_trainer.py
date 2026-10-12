import os
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # Demand-set / held-out eval (optional; default off for existing configs)
        world_param = Registry.mapping['world_mapping']['setting'].param
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        self.demand_set = (
            world_param.get('demand_set')
            or world_param.get('demand_bag')
            or []
        )
        self.demand_heldout = world_param.get('demand_heldout') or []
        self.demand_train_file = world_param.get('demand_train_file')
        self.heldout_eval_every = int(trainer_param.get('heldout_eval_every') or 0)
        # Adaptive episode budget (L1/L2: cycle-mean median wait). Missing keys = run full episodes.
        self.min_episodes = int(trainer_param.get('min_episodes') or 0)
        self.early_stop = bool(trainer_param.get('early_stop', False))
        self.early_stop_patience = int(trainer_param.get('early_stop_patience') or 0)
        self.early_stop_min_delta = float(trainer_param.get('early_stop_min_delta') or 0.0)
        self.early_stop_metric = str(trainer_param.get('early_stop_metric') or 'test').lower()
        self.early_stop_abs_floor = float(trainer_param.get('early_stop_abs_floor') or 0.0)
        mode = str(trainer_param.get('early_stop_mode') or 'auto').lower()
        cycle_len = len(self.demand_set) if len(self.demand_set) > 1 else 10
        if mode in ('', 'auto'):
            if self.early_stop_metric == 'wait' and len(self.demand_set) > 1:
                mode = 'cycle_mean'
            else:
                mode = 'off'
        self.early_stop_mode = mode
        self.early_stop_cycle_len = cycle_len
        self._cycle_mean_stopper = None
        # Per-vehicle trip metrics (SUMO): final test (+ best checkpoint for RL)
        cmd = Registry.mapping['command_mapping']['setting'].param
        self.save_trip_metrics = bool(trainer_param.get('save_trip_metrics', True))
        if cmd.get('no_trip_metrics'):
            self.save_trip_metrics = False
        self._last_trip_records = None
        self.train_model = bool(
            Registry.mapping['model_mapping']['setting'].param.get('train_model', False)
        )
        self.best_episode = None
        self.best_att = None
        self.best_metric_source = None  # 'test' | 'heldout'
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def _trip_metrics_supported(self):
        cmd = Registry.mapping['command_mapping']['setting'].param
        if cmd.get('world') != 'sumo':
            return False
        return hasattr(self.world, 'eng') and hasattr(getattr(self.world, 'eng', None), 'vehicle')

    def _trip_metrics_dir(self):
        return os.path.join(
            Registry.mapping['logger_mapping']['path'].path,
            Registry.mapping['logger_mapping']['setting'].param['log_dir'],
        )

    def _write_trip_metrics(self, records, stem='new_metrics', meta_extra=None):
        from utils.trip_metrics import write_trip_metrics
        cmd = Registry.mapping['command_mapping']['setting'].param
        extra = {
            'agent': cmd.get('agent'),
            'network': cmd.get('network'),
            'world': cmd.get('world'),
            'prefix': cmd.get('prefix'),
            'seed': cmd.get('seed'),
            'avg_travel_time_metric': float(self.metric.real_average_travel_time()),
            'throughput_metric': int(self.metric.throughput()),
            'best_episode': self.best_episode,
            'best_att': self.best_att,
            'best_metric_source': self.best_metric_source,
        }
        if meta_extra:
            extra.update(meta_extra)
        csv_path, meta_path, meta = write_trip_metrics(
            records, self._trip_metrics_dir(), meta_extra=extra, stem=stem
        )
        self.logger.info(
            "Saved %d vehicle trip records to %s (ATT=%.2fs, completion=%.1f%%)",
            len(records),
            csv_path,
            meta.get('mean_travel_time_s') or 0.0,
            (meta.get('completion_rate') or 0.0) * 100.0,
        )
        return csv_path, meta_path, meta

    def _maybe_update_best(self, e, att, source='test'):
        '''Track lowest eval ATT during RL training and snapshot weights to best_*.pt.'''
        if not self.train_model or att is None:
            return
        att = float(att)
        if self.best_att is not None and att >= self.best_att:
            return
        self.best_att = att
        self.best_episode = int(e)
        self.best_metric_source = source
        try:
            [ag.save_model(e='best') for ag in self.agents]
            self.logger.info(
                "New best %s ATT=%.4f at episode %s (saved model/best_*.pt)",
                source, att, e,
            )
        except Exception as exc:
            self.logger.warning("Failed to save best checkpoint at episode %s: %s", e, exc)

    def _load_best_checkpoint(self):
        '''Load best_*.pt if present; otherwise fall back to episode-numbered file.'''
        try:
            [ag.load_model('best') for ag in self.agents]
            return 'best'
        except Exception as exc:
            self.logger.warning(
                "Could not load model/best_*.pt (%s); trying episode %s",
                exc, self.best_episode,
            )
            [ag.load_model(self.best_episode) for ag in self.agents]
            return self.best_episode

    def _save_best_trip_metrics(self):
        '''Re-evaluate the best RL checkpoint and write new_metrics_best*.'''
        if not (self.train_model and self.save_trip_metrics and self._trip_metrics_supported()):
            return
        if self.best_episode is None:
            self.logger.info(
                "No best checkpoint tracked (need test_when_train or heldout_eval_every); "
                "skipping new_metrics_best*."
            )
            return
        loaded = self._load_best_checkpoint()
        meta_common = {
            'checkpoint': 'best',
            'checkpoint_episode': self.best_episode,
            'checkpoint_loaded': loaded,
            'best_metric_source': self.best_metric_source,
            'best_att_during_train': self.best_att,
        }
        if self.demand_heldout and hasattr(self.world, 'set_route_file'):
            self.heldout_eval(
                self.best_episode,
                save_trip_metrics=True,
                metrics_stem_prefix='new_metrics_best_hold',
                update_best=False,
                checkpoint_meta=meta_common,
            )
            return
        if self.demand_train_file and hasattr(self.world, 'set_route_file'):
            self.world.set_route_file(self.demand_train_file)
        att = self._run_eval_episode(collect_trip_metrics=True)
        self.logger.info(
            "Best-checkpoint Travel Time is %.4f (episode %s, source=%s)",
            att, self.best_episode, self.best_metric_source,
        )
        if self._last_trip_records is not None:
            meta_common = dict(meta_common)
            meta_common['eval_att'] = float(att)
            self._write_trip_metrics(
                self._last_trip_records,
                stem='new_metrics_best',
                meta_extra=meta_common,
            )

    def _collect_actions(self, obs, phases, test=True):
        '''Gather actions for all agents; use batch path when the agent provides one.'''
        first = self.agents[0]
        if hasattr(first, "get_actions_batch"):
            return np.stack(first.get_actions_batch(self.agents, obs, phases, test=test))
        return np.stack(
            [ag.get_action(obs[idx], phases[idx], test=test) for idx, ag in enumerate(self.agents)]
        )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
            # real delay needs per-step vehicle trajectories; gate is off by default.
            if hasattr(self.world, 'update_vehicle_trajectory'):
                self.world.update_vehicle_trajectory = True
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        model_name = Registry.mapping['model_mapping']['setting'].param['name']
        agent = Registry.mapping['model_mapping'][model_name](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][model_name](self.world, i))

        # pass device from trainer to all agents for GPU support
        for ag in self.agents:
            ag.to_device(self.device)
        print(f"[Device] Moved {len(self.agents)} agent(s) to {self.device}")

        # for magd agents should share information
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)
            # re-apply device after link_agents builds models
            for ag in self.agents:
                ag.to_device(self.device)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s). Stops at episodes (max) or earlier when early_stop
        is enabled (cycle-mean median wait on rotating demand).

        :param: None
        :return: None
        '''
        total_decision_num = 0
        flush = 0
        self._cycle_mean_stopper = None
        if (self.early_stop_metric == 'wait' or self.early_stop_mode == 'cycle_mean') and hasattr(
                self.world, 'track_trip_wait'):
            self.world.track_trip_wait = True
        if self.early_stop and self.early_stop_mode == 'cycle_mean':
            from common.early_stop import CycleMeanStopper
            self._cycle_mean_stopper = CycleMeanStopper(
                cycle_len=self.early_stop_cycle_len,
                patience=self.early_stop_patience,
                min_delta=self.early_stop_min_delta,
                abs_floor=self.early_stop_abs_floor,
                min_episodes=self.min_episodes,
            )
            self.logger.info(
                "early_stop enabled: mode={}, min_episodes={}, max_episodes={}, "
                "patience={}, min_delta={:.4f}, abs_floor={:.2f}, cycle_len={}, metric={}".format(
                    self.early_stop_mode, self.min_episodes, self.episodes,
                    self.early_stop_patience, self.early_stop_min_delta,
                    self.early_stop_abs_floor, self.early_stop_cycle_len,
                    self.early_stop_metric)
            )
        for e in range(self.episodes):
            # TODO: check this reset agent
            self._select_train_demand(e)
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = self._collect_actions(last_obs, last_phase, test=False)
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for t in range(self.action_interval):
                        # Intermediate observations are discarded; only the last obs of
                        # the interval is stored in the replay buffer / next state.
                        obs, rewards, dones, _ = self.env.step(
                            actions.flatten(),
                            collect_obs=(t == self.action_interval - 1),
                        )
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            
            train_att = self.metric.real_average_travel_time()
            self.writeLog("TRAIN", e, train_att,\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, train_att))
            median_wait = None
            if hasattr(self.world, 'get_median_waiting_time'):
                median_wait = self.world.get_median_waiting_time()
                n_wait = len(getattr(self.world, 'departed_wait_s', {}) or {})
                if median_wait is not None:
                    self.logger.info(
                        "episode:{}/{}, median wait:{} (n={})".format(
                            e, self.episodes, median_wait, n_wait)
                    )
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            if self.test_when_train:
                self.train_test(e)
            if self.heldout_eval_every > 0 and self.demand_heldout and (e % self.heldout_eval_every == 0):
                self.heldout_eval(e)
            if self._cycle_mean_stopper is not None:
                stop, decision = self._cycle_mean_stopper.add(median_wait)
                if decision is not None:
                    self.logger.info(
                        "cycle_mean cycle={} m_prev={:.2f} m_curr={:.2f} rel={:+.4f} "
                        "delta={:+.2f} flat={} improved={} patience={}/{}".format(
                            decision["cycle"], decision["mean_prev"], decision["mean_curr"],
                            decision["rel"], decision["delta"], decision["flat"],
                            decision["improved"], decision["cycles_without_improve"],
                            self.early_stop_patience)
                    )
                if self.early_stop and stop:
                    self.logger.info(
                        "early_stop at episode {}/{} (cycle_mean wait, m={:.2f}, "
                        "best ATT={} at episode {}, source={})".format(
                            e, self.episodes,
                            decision["mean_curr"] if decision else float("nan"),
                            "{:.2f}".format(self.best_att) if self.best_att is not None else "n/a",
                            self.best_episode, self.best_metric_source)
                    )
                    break
        if self.best_episode is not None:
            try:
                self._load_best_checkpoint()
                self.logger.info(
                    "reloaded best checkpoint from episode {} (ATT={:.2f}, source={})".format(
                        self.best_episode, self.best_att, self.best_metric_source)
                )
            except Exception as exc:
                self.logger.warning("could not reload best checkpoint: {}".format(exc))
        [ag.save_model(e=self.episodes) for ag in self.agents]

    def _select_train_demand(self, e):
        '''Pick route file for this training episode (fixed or demand-set rotation).'''
        if not hasattr(self.world, 'set_route_file'):
            return
        if self.demand_set:
            route = self.demand_set[e % len(self.demand_set)]
            self.world.set_route_file(route)
            self.logger.info("episode:{}/{}, train_demand:{}".format(e, self.episodes, route))
        elif self.demand_train_file:
            self.world.set_route_file(self.demand_train_file)

    def _run_eval_episode(self, collect_trip_metrics=False):
        '''Greedy rollout for test_steps; returns real avg travel time.

        When collect_trip_metrics is True (SUMO), fills self._last_trip_records.
        '''
        obs = self.env.reset()
        self.metric.clear()
        self._last_trip_records = None
        for a in self.agents:
            a.reset()
        tracker = None
        if collect_trip_metrics and self._trip_metrics_supported():
            from utils.trip_metrics import TripMetricsTracker
            tracker = TripMetricsTracker(self.world)
        dones = [False]
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = self._collect_actions(obs, phases, test=True)
                rewards_list = []
                for t in range(self.action_interval):
                    if tracker is not None:
                        tracker.before_step()
                    obs, rewards, dones, _ = self.env.step(
                        actions.flatten(),
                        collect_obs=(t == self.action_interval - 1),
                    )
                    if tracker is not None:
                        tracker.after_step()
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)
                self.metric.update(rewards)
            if all(dones):
                break
        if tracker is not None:
            self._last_trip_records = tracker.finalize()
        return self.metric.real_average_travel_time()

    def heldout_eval(
        self,
        e,
        save_trip_metrics=False,
        metrics_stem_prefix='new_metrics_hold',
        update_best=True,
        checkpoint_meta=None,
    ):
        '''Evaluate (no learning) on held-out demand files and log mean ATT.'''
        atts = []
        for idx, route in enumerate(self.demand_heldout):
            if hasattr(self.world, 'set_route_file'):
                self.world.set_route_file(route)
            att = self._run_eval_episode(collect_trip_metrics=save_trip_metrics)
            atts.append(att)
            self.logger.info(
                "HELDOUT episode:{}/{}, demand:{}, travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                    e, self.episodes, route, att, self.metric.rewards(),
                    self.metric.queue(), self.metric.delay(), int(self.metric.throughput()))
            )
            self.writeLog("HELDOUT", e, att, 100, self.metric.rewards(),
                          self.metric.queue(), self.metric.delay(), self.metric.throughput())
            if hasattr(self.world, 'get_median_waiting_time'):
                mw = self.world.get_median_waiting_time()
                if mw is not None:
                    self.logger.info(
                        "HELDOUT episode:{}/{}, demand:{}, median wait:{}".format(
                            e, self.episodes, route, mw)
                    )
            if save_trip_metrics and self._last_trip_records is not None:
                meta_extra = {
                    'demand_file': route,
                    'heldout_index': idx,
                    'episode': e,
                    'checkpoint': (checkpoint_meta or {}).get('checkpoint', 'final'),
                }
                if checkpoint_meta:
                    meta_extra.update(checkpoint_meta)
                self._write_trip_metrics(
                    self._last_trip_records,
                    stem=f"{metrics_stem_prefix}_{idx:02d}",
                    meta_extra=meta_extra,
                )
        mean_att = float(np.mean(atts)) if atts else 0.0
        self.logger.info(
            "HELDOUT_MEAN episode:{}/{}, travel time:{} (n={})".format(
                e, self.episodes, mean_att, len(atts))
        )
        self.writeLog("HELDOUT_MEAN", e, mean_att, 100, 0, 0, 0, 0)
        if update_best:
            self._maybe_update_best(e, mean_att, source='heldout')
        return mean_att

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        att = self._run_eval_episode()
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, att, self.metric.rewards(),\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, att,\
            100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput())
        if getattr(self.env.world, 'crossing_proxy_ctrl', None) is not None:
            self.env.world.crossing_proxy_ctrl.log_summary()
        self._maybe_update_best(e, att, source='test')
        return att

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        collect = bool(self.save_trip_metrics)
        if self.demand_heldout and hasattr(self.world, 'set_route_file'):
            self.heldout_eval(
                self.episodes,
                save_trip_metrics=collect,
                metrics_stem_prefix='new_metrics_hold',
                update_best=False,
                checkpoint_meta={'checkpoint': 'final'},
            )
            self._save_best_trip_metrics()
            return self.metric
        if self.demand_train_file and hasattr(self.world, 'set_route_file'):
            self.world.set_route_file(self.demand_train_file)
        att = self._run_eval_episode(collect_trip_metrics=collect)
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (att, \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        if getattr(self.env.world, 'crossing_proxy_ctrl', None) is not None:
            self.env.world.crossing_proxy_ctrl.log_summary()
        if collect and self._last_trip_records is not None:
            self._write_trip_metrics(
                self._last_trip_records,
                stem='new_metrics',
                meta_extra={'checkpoint': 'final', 'eval_att': float(att)},
            )
        self._save_best_trip_metrics()
        return self.metric

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

