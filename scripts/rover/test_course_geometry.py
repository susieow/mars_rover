"""Check actual scene geometry and reject incorrect collection declarations."""
import unittest
from pathlib import Path
import mujoco
import course
from verify_dataset import inspect_course

class CourseGeometryTests(unittest.TestCase):
    def model(self, height=.15, stairs=15):
        path=course.build_course(height,stairs,height,4.,nosing=0.,contact_timeconst=.004)
        return mujoco.MjModel.from_xml_path(str(path))

    def test_fifteen_risers_for_both_requested_heights(self):
        for height in (.15,.20):
            with self.subTest(height=height):
                result=inspect_course(self.model(height),dict(stairs=15,height=height,depth=height))
                self.assertEqual(result["ascending_risers"],15)
                self.assertEqual(result["descending_risers_including_ground"],15)
                self.assertAlmostEqual(result["platform_top_z"],15*height)

    def test_existing_twelve_riser_seed_remains_valid(self):
        path=Path("/home/agilex/wzy/rover/datasets/45deg_v3/episodes/seed1000_h04_base/scene.xml")
        result=inspect_course(mujoco.MjModel.from_xml_path(str(path)),dict(stairs=12,height=.04,depth=.04))
        self.assertEqual(result["ascending_risers"],12)

    def test_declaration_cannot_mislabel_geometry(self):
        m=self.model()
        for params in (dict(stairs=12,height=.15,depth=.15),
                       dict(stairs=15,height=.20,depth=.15),
                       dict(stairs=15,height=.15,depth=.20),
                       dict(stairs=15.0,height=.15,depth=.15)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                inspect_course(m,params)

    def test_nonuniform_riser_is_rejected(self):
        m=self.model()
        gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"up7")
        m.geom_pos[gid,2]+=.005
        m.geom_size[gid,2]+=.005
        with self.assertRaisesRegex(ValueError,"incorrect riser height"):
            inspect_course(m)

    def test_gap_is_rejected(self):
        m=self.model()
        gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"down7")
        m.geom_pos[gid,0]+=.001
        with self.assertRaisesRegex(ValueError,"overlap or gap"):
            inspect_course(m)

    def test_missing_step_is_rejected(self):
        p=course.build_course(.15,15,.15,4.,nosing=0.,contact_timeconst=.004)
        xml=p.read_text().replace('name="up7"','name="up99"')
        xml=xml.replace('file="meshes/','file="'+str(course.ROOT)+'/meshes/')
        with self.assertRaisesRegex(ValueError,"consecutive"):
            inspect_course(mujoco.MjModel.from_xml_string(xml))

if __name__=="__main__":
    unittest.main(verbosity=2)
