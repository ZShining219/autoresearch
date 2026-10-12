from .base import BaseAgent
from .rl_agent import RLAgent

# Always-available baselines
from .maxpressure import MaxPressureAgent
from .fixedtime import FixedTimeAgent
from .sotl import SOTLAgent

# Generic LLM TSC controller; backends/deps loaded lazily inside the agent
try:
	from .llm_tsc import LLMTSCAgent
except Exception as e:
	print(f"Warning: Failed to import LLM TSC agent: {e}")
	LLMTSCAgent = None

# Native OpenRouter Decisions controller; HTTP dependencies load on first use.
try:
	from .jev import JEVAgent
except ImportError as e:
	print(f"Warning: Failed to import JEV agent: {e}")
	JEVAgent = None

# Optional RL agents; import lazily so missing deps (e.g., torch_scatter) don't block baselines
try:
	from .colight import CoLightAgent
	from .colight_rich import CoLightRichAgent
except Exception as e:
	print(f"Warning: Failed to import CoLightAgent: {e}")
	CoLightAgent = None
	CoLightRichAgent = None

try:
	from .dqn import DQNAgent
	from .dqn_rich import DQNRichAgent
except ModuleNotFoundError:
	DQNAgent = None
	DQNRichAgent = None

try:
	from .frap import FRAP_DQNAgent
except ModuleNotFoundError:
	FRAP_DQNAgent = None

try:
	from .ppo_pfrl import IPPO_pfrl
except ModuleNotFoundError:
	IPPO_pfrl = None

# from .maddpg import MADDPGAgent
try:
	from .maddpg_v2 import MADDPGAgent
except ModuleNotFoundError:
	MADDPGAgent = None

try:
	from .magd import MAGDAgent
except ModuleNotFoundError:
	MAGDAgent = None

try:
	from .presslight import PressLightAgent
	from .presslight_rich import PressLightRichAgent
except ModuleNotFoundError:
	PressLightAgent = None
	PressLightRichAgent = None

try:
	from .mplight import MPLightAgent
except ModuleNotFoundError:
	MPLightAgent = None
