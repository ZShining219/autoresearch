"""Hub-OD hetero rewrite: trucks must appear on pkw demand files."""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from world.world_sumo import rewrite_route_file_hetero_mix


class HeteroHubOdRewriteTest(unittest.TestCase):
    def test_pkw_file_becomes_80_20_car_truck(self):
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<routes>
  <vType id="pkw" length="5.0" accel="2.0"/>
  <vehicle id="0" type="pkw" depart="0"/>
  <vehicle id="1" type="pkw" depart="1"/>
  <vehicle id="2" type="pkw" depart="2"/>
  <vehicle id="3" type="pkw" depart="3"/>
  <vehicle id="4" type="pkw" depart="4"/>
</routes>
'''
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'hold_00.rou.xml')
            with open(src, 'w') as f:
                f.write(xml)
            dst = rewrite_route_file_hetero_mix(src)
            self.assertNotEqual(dst, src)
            self.assertIn('.hetero_mix', dst)
            root = ET.parse(dst).getroot()
            self.assertEqual(root.findall('vType'), [])
            types = [v.get('type') for v in root.findall('vehicle')]
            self.assertEqual(types.count('truck'), 1)
            self.assertEqual(types.count('car'), 4)
            self.assertNotIn('pkw', types)

    def test_existing_hetero_file_is_unchanged(self):
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<routes>
  <vehicle id="0" type="car" depart="0"/>
  <vehicle id="1" type="truck" depart="1"/>
</routes>
'''
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'grid4x4_hetero.rou.xml')
            with open(src, 'w') as f:
                f.write(xml)
            dst = rewrite_route_file_hetero_mix(src)
            self.assertEqual(dst, src)
