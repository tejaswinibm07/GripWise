"""
GripWise Comprehensive Automated Test Suite
===========================================
Validates:
1. Dataset Registry integrity & metadata segregation.
2. Leakage-free split validation (Recording-level & Subject-independent splits).
3. Preprocessing adapter data contracts and feature extraction.
4. ML Models A, B1, B2, C, D training, evaluation, and inference.
5. Graceful handling of missing files and error boundaries.
6. Core simulation and digital twin state-machine regression.

Runs natively with:
  python tests/test_pipeline_research.py
or:
  python -m unittest tests/test_pipeline_research.py
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_registry import (
    DatasetRegistry,
    DataNature,
    MLTask
)
from dataset_preprocessing import (
    BristolSlipAdapter,
    IntelliManSlipAdapter,
    NinaProEMGAdapter
)
from slip_model import (
    IntelliManPVDFSlipClassifier,
    BristolTacTipSlipClassifier
)
from grasp_model import ContactGraspStateClassifier
from emg_model import EMGIntentionClassifier
from prediction import ForcePredictor, compute_baseline_safe_force
from simulation import HandState, ObjectSpec, STANDARD_OBJECTS, ProstheticHandPhysics
from controller import AdaptiveGripController
from hardware import (
    SensorPacket,
    PacketParser,
    SimulatedSensorInterface,
    MockSerialSensorInterface,
    RecordedReplaySensorInterface,
)
from virtual_hand import generate_hand_svg
from demo_scenario import execute_end_to_end_demonstration, DemonstrationExecutionSummary
from personalization import PersonalizationManager


class TestGripWiseResearchPipeline(unittest.TestCase):
    """Automated test suite covering data pipeline, models, and simulation."""

    # ========================================================================
    # 1. DATASET REGISTRY TESTS
    # ========================================================================

    def test_01_registry_all_datasets_registered(self):
        """Verify that all target datasets are registered with complete metadata."""
        registry = DatasetRegistry()
        datasets = registry.list_datasets()
        catalog = {d.key: d for d in datasets}
        
        expected_datasets = [
            "bristol_slip",
            "intelliman_slip",
            "intelliman_texture",
            "ninapro_emg",
            "gripwise_force_synthetic"
        ]
        for ds_id in expected_datasets:
            self.assertIn(ds_id, catalog, f"Dataset {ds_id} missing from catalog")
            meta = catalog[ds_id]
            self.assertTrue(bool(meta.name), f"{ds_id} has empty name")
            self.assertTrue(bool(meta.citation), f"{ds_id} has empty citation")
            self.assertTrue(bool(meta.license_type), f"{ds_id} has empty license_type")
            self.assertTrue(bool(meta.source_url), f"{ds_id} has empty source_url")
            self.assertIn(meta.data_nature, [DataNature.REAL_RESEARCH_DATA, DataNature.SYNTHETIC_DATA])

    def test_02_registry_strict_nature_segregation(self):
        """Ensure real research datasets are never mislabeled as synthetic and vice versa."""
        registry = DatasetRegistry()
        
        real_datasets = ["bristol_slip", "intelliman_slip", "intelliman_texture", "ninapro_emg"]
        for ds_id in real_datasets:
            meta = registry.get_dataset(ds_id)
            self.assertIsNotNone(meta, f"Dataset {ds_id} not found in registry")
            self.assertEqual(meta.data_nature, DataNature.REAL_RESEARCH_DATA, f"{ds_id} must be REAL_RESEARCH_DATA")
        
        synth_meta = registry.get_dataset("gripwise_force_synthetic")
        self.assertIsNotNone(synth_meta, "gripwise_force_synthetic not found in registry")
        self.assertEqual(synth_meta.data_nature, DataNature.SYNTHETIC_DATA, "gripwise_force_synthetic must be SYNTHETIC_DATA")

    def test_03_registry_status_methods(self):
        """Verify registry summary table generation."""
        registry = DatasetRegistry()
        summary_rows = registry.get_summary_table()
        self.assertIsInstance(summary_rows, list)
        self.assertGreater(len(summary_rows), 0)
        df_status = pd.DataFrame(summary_rows)
        self.assertIn("Key", df_status.columns)
        self.assertIn("Download Status", df_status.columns)
        self.assertIn("Processed Status", df_status.columns)

    # ========================================================================
    # 2. DATA LEAKAGE PREVENTION & PREPROCESSING TESTS
    # ========================================================================

    def test_04_intelliman_leakage_free_split(self):
        """
        Critical Test: Ensure IntelliMan split is strictly by recording/file ID,
        guaranteeing that train and test window frames NEVER share recordings.
        """
        train_path = os.path.abspath("data/processed/intelliman_slip/intelliman_slip_train.csv")
        test_path = os.path.abspath("data/processed/intelliman_slip/intelliman_slip_test.csv")
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            adapter = IntelliManSlipAdapter()
            if not adapter.meta.is_downloaded():
                self.skipTest("IntelliMan data not downloaded; skipping test")
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
        self.assertFalse(df_train.empty)
        self.assertFalse(df_test.empty)
        
        train_recordings = set(df_train["trial_file"].unique())
        test_recordings = set(df_test["trial_file"].unique())
        
        overlap = train_recordings.intersection(test_recordings)
        self.assertEqual(len(overlap), 0, f"DATA LEAKAGE DETECTED: Recordings {overlap} present in both train and test!")
        
        # Check feature sanity
        self.assertIn("is_slip", df_train.columns)
        self.assertIn("global_vibration_mean_std", df_train.columns)
        self.assertFalse(df_train.isna().any().any(), "Train data contains NaNs")
        self.assertFalse(df_test.isna().any().any(), "Test data contains NaNs")

    def test_05_bristol_leakage_free_split(self):
        """
        Critical Test: Ensure Bristol TacTip split uses distinct runs (disjoint sets of run MAT files).
        """
        train_path = os.path.abspath("data/processed/bristol_slip/bristol_slip_train.csv")
        test_path = os.path.abspath("data/processed/bristol_slip/bristol_slip_test.csv")
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            adapter = BristolSlipAdapter()
            if not adapter.meta.is_downloaded():
                self.skipTest("Bristol data not downloaded; skipping test")
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
        self.assertFalse(df_train.empty)
        self.assertFalse(df_test.empty)
        
        # Bristol official benchmark: train folder vs test folder
        train_runs = set(df_train["split"].astype(str) + "/" + df_train["recording"].astype(str))
        test_runs = set(df_test["split"].astype(str) + "/" + df_test["recording"].astype(str))
        
        overlap = train_runs.intersection(test_runs)
        self.assertEqual(len(overlap), 0, f"DATA LEAKAGE DETECTED: Bristol runs {overlap} present in both train and test!")
        self.assertTrue(all(df_train["split"] == "train"))
        self.assertTrue(all(df_test["split"] == "test"))
        
        # Feature columns sanity
        self.assertIn("is_slip", df_train.columns)
        self.assertIn("mean_pin_speed", df_train.columns)
        self.assertIn("spatial_coherence", df_train.columns)

    def test_06_ninapro_subject_independent_split(self):
        """
        Critical Test: Verify NinaPro EMG splits by subject ID to test subject generalization.
        """
        train_path = os.path.abspath("data/processed/ninapro_emg/ninapro_emg_train.csv")
        test_path = os.path.abspath("data/processed/ninapro_emg/ninapro_emg_test.csv")
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            adapter = NinaProEMGAdapter()
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
        self.assertFalse(df_train.empty)
        self.assertFalse(df_test.empty)
        
        train_subjects = set(df_train["subject_id"].unique())
        test_subjects = set(df_test["subject_id"].unique())
        
        overlap = train_subjects.intersection(test_subjects)
        self.assertEqual(len(overlap), 0, f"DATA LEAKAGE DETECTED: Subjects {overlap} present in both train and test!")
        self.assertIn("gesture_id", df_train.columns)

    # ========================================================================
    # 3. MODEL TRAINING & INFERENCE INTEGRITY TESTS
    # ========================================================================

    def test_07_model_a_force_predictor(self):
        """Test Model A: Grip force regression model and baseline."""
        predictor = ForcePredictor()
        self.assertIsNotNone(predictor.model)
        
        cup = STANDARD_OBJECTS["Cup"]
        pred = predictor.predict(
            obj=cup,
            user_safety_margin=1.25,
            algorithm="Random Forest (ML)"
        )
        self.assertGreater(pred.recommended_target_n, 0)
        self.assertLessEqual(pred.recommended_target_n, cup.fragility_limit_n)
        self.assertIn("Random Forest", pred.model_type)
        self.assertGreater(len(pred.feature_importances), 0)

    def test_08_model_b1_intelliman_pvdf_slip_classifier(self):
        """Test Model B1: IntelliMan PVDF slip detection classifier."""
        clf = IntelliManPVDFSlipClassifier()
        if clf.model is None:
            metrics = clf.train_and_evaluate()
        else:
            metrics = clf.metrics
            
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.accuracy, 0.60)
        self.assertGreater(metrics.roc_auc, 0.70)
        
        # Test inference on single feature vector
        sample_vec = np.zeros(len(clf.FEATURE_COLS))
        pred, prob = clf.predict(sample_vec)
        self.assertIn(pred, [0, 1])
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_09_model_b2_bristol_tactip_slip_classifier(self):
        """Test Model B2: Bristol TacTip optical slip detection classifier."""
        clf = BristolTacTipSlipClassifier()
        if clf.model is None:
            metrics = clf.train_and_evaluate()
        else:
            metrics = clf.metrics
            
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.accuracy, 0.70)
        self.assertGreater(metrics.roc_auc, 0.75)
        
        # Test inference
        sample_vec = np.array([0.05, 0.02, 0.10, 0.01, 0.85, 0.05])
        X_vec = sample_vec.reshape(1, -1)
        pred = int(clf.model.predict(X_vec)[0])
        prob = float(clf.model.predict_proba(X_vec)[0, 1])
        self.assertIn(pred, [0, 1])
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_10_model_c_contact_grasp_state_classifier(self):
        """Test Model C: Contact/Slip state 3-class classification."""
        clf = ContactGraspStateClassifier()
        if clf.model is None:
            metrics = clf.train_and_evaluate()
        else:
            metrics = clf.metrics
            
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.accuracy, 0.60)
        self.assertGreater(metrics.macro_f1, 0.50)
        self.assertEqual(len(metrics.class_names), 3)

    def test_11_model_d_emg_intention_classifier(self):
        """Test Model D: sEMG Movement Intention Classifier."""
        clf = EMGIntentionClassifier()
        if clf.model is None:
            metrics = clf.train_and_evaluate()
        else:
            metrics = clf.metrics
            
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.accuracy, 0.60)
        self.assertEqual(len(metrics.gestures), 5)

    # ========================================================================
    # 4. DIGITAL TWIN & SIMULATION REGRESSION TESTS
    # ========================================================================

    def test_12_simulation_physics(self):
        """Verify simulation physics calculations, slip risk, and kinematics."""
        cup = STANDARD_OBJECTS["Cup"]
        physics = ProstheticHandPhysics(cup)
        self.assertEqual(physics.state, HandState.OPEN)
        
        f_min = physics.calculate_theoretical_min_slip_force()
        self.assertGreater(f_min, 0.5)
        
        # Flexion kinematics
        flex = physics.get_kinematic_flexion(0.5)
        self.assertGreater(flex.thumb, 0.0)
        self.assertLessEqual(flex.thumb, 1.0)
        
        # Slip risk when below min force
        risk, is_slipping = physics.evaluate_slip_risk(0.4)
        self.assertTrue(is_slipping)
        self.assertGreater(risk, 1.0)

    def test_13_controller_adaptive_recovery(self):
        """Verify adaptive closed-loop controller reacts to micro-slip events."""
        cup = STANDARD_OBJECTS["Cup"]
        sensor = SimulatedSensorInterface()
        controller = AdaptiveGripController(obj=cup, base_target=3.5)
        
        controller.start_grasp_sequence(sensor)
        self.assertEqual(controller.state, HandState.CLOSING)
        
        # Step through grasp until grasped
        for _ in range(15):
            pt = controller.step(sensor)
        
        self.assertIn(controller.state, [HandState.GRASPING, HandState.STABLE_GRIP])
        target_before = controller.current_target
        
        # Inject slip event
        controller.trigger_slip_event(sensor)
        pt_after_slip = controller.step(sensor)
        self.assertGreaterEqual(controller.current_target, target_before)

    def test_14_virtual_hand_svg_generation(self):
        """Verify virtual hand returns valid non-empty SVG XML string."""
        cup = STANDARD_OBJECTS["Cup"]
        svg = generate_hand_svg(
            state=HandState.STABLE_GRIP,
            flexion_ratio=0.8,
            current_force=3.2,
            target_force=3.5,
            obj=cup
        )
        self.assertIsInstance(svg, str)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("Ceramic Cup", svg)

    # ========================================================================
    # 5. END-TO-END DEMONSTRATION & COMPARISON TESTS
    # ========================================================================

    def test_15_end_to_end_demonstration_scenario(self):
        """Verify the full 8-stage end-to-end AI demonstration sequence."""
        pm = PersonalizationManager()
        pred = ForcePredictor()
        initial_trials_count = len(pm.trials)

        summary = execute_end_to_end_demonstration(
            object_key="Cup",
            profile_name="Alex (Standard)",
            predictor=pred,
            personalization_mgr=pm
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.outcome, "ADAPTED_SUCCESS")
        self.assertTrue(summary.slip_injected)
        self.assertGreater(summary.adapted_target_force_n, summary.initial_predicted_force_n)
        self.assertGreater(summary.fragility_headroom_n, 0.0)
        self.assertEqual(len(summary.steps_log), 8)
        self.assertEqual(len(pm.trials), initial_trials_count + 1)
        self.assertEqual(pm.trials[0].trial_id, summary.trial_id_logged)

    def test_16_offline_research_inference_pvdf(self):
        """Verify offline inference using actual held-out PVDF test sample."""
        test_csv_path = os.path.abspath("data/processed/intelliman_slip/intelliman_slip_test.csv")
        if not os.path.exists(test_csv_path):
            self.skipTest("IntelliMan test dataset not present")

        df_test = pd.read_csv(test_csv_path)
        clf = IntelliManPVDFSlipClassifier()
        if clf.model is None:
            clf.train_and_evaluate()

        sample_row = df_test.iloc[0]
        feat_cols = [c for c in clf.FEATURE_COLS if c in sample_row.index]
        sample_vec = sample_row[feat_cols].values

        pred_class, slip_prob = clf.predict(sample_vec)
        self.assertIn(pred_class, [0, 1])
        self.assertTrue(0.0 <= slip_prob <= 1.0)

    def test_17_offline_research_inference_tactip(self):
        """Verify offline inference using actual held-out Bristol TacTip test sample."""
        test_csv_path = os.path.abspath("data/processed/bristol_slip/bristol_slip_test.csv")
        if not os.path.exists(test_csv_path):
            self.skipTest("Bristol test dataset not present")

        df_test = pd.read_csv(test_csv_path)
        clf = BristolTacTipSlipClassifier()
        if clf.model is None:
            clf.train_and_evaluate()

        sample_row = df_test.iloc[0]
        sample_vec = sample_row[clf.FEATURE_COLS].values.reshape(1, -1)
        pred_class = int(clf.model.predict(sample_vec)[0])
        slip_prob = float(clf.model.predict_proba(sample_vec)[0, 1])

        self.assertIn(pred_class, [0, 1])
        self.assertTrue(0.0 <= slip_prob <= 1.0)

    def test_18_force_model_vs_baseline_comparison(self):
        """Verify comparison across all 5 standard objects between baseline and Model A."""
        predictor = ForcePredictor()
        for name, obj in STANDARD_OBJECTS.items():
            base_f = compute_baseline_safe_force(
                mass_g=obj.mass_g,
                friction_coeff=obj.friction_coeff,
                compliance_mm_per_n=obj.compliance_mm_per_n,
                safety_margin_factor=1.35
            )
            pred_res = predictor.predict(obj=obj, user_safety_margin=1.35, algorithm="Random Forest (ML)")
            rf_f = pred_res.recommended_target_n

            self.assertGreater(base_f, 0.0)
            self.assertGreater(rf_f, 0.0)
            self.assertLessEqual(rf_f, obj.fragility_limit_n)
            self.assertLessEqual(base_f, obj.fragility_limit_n * 1.5)

    # ========================================================================
    # 7. HARDWARE WIRE PROTOCOL, PACKET PARSER & TEST BENCH TESTS
    # ========================================================================

    def test_19_packet_parser_valid_telemetry(self):
        """Verify PacketParser parses standard valid ASCII telemetry lines."""
        # 1. Standard stable grasp packet
        line_stable = "3.85,0,1.54,0.85,0.77,0.42,0.27,245.0\n"
        pkt, success = PacketParser.parse_raw_packet(line_stable, source_tag="ESP32_UART")
        self.assertTrue(success)
        self.assertTrue(pkt.is_valid)
        self.assertFalse(pkt.is_stale)
        self.assertEqual(pkt.parse_error, "")
        self.assertAlmostEqual(pkt.total_force, 3.85, places=2)
        self.assertFalse(pkt.slip_detected)
        self.assertAlmostEqual(pkt.thumb_force, 1.54, places=2)
        self.assertAlmostEqual(pkt.pinky_force, 0.27, places=2)
        self.assertAlmostEqual(pkt.motor_current_ma, 245.0, places=1)
        self.assertEqual(pkt.source, "ESP32_UART")

        # 2. Slip event packet
        line_slip = "4.50,1,1.80,0.99,0.90,0.50,0.31,310.5\r\n"
        pkt_slip, success_slip = PacketParser.parse_raw_packet(line_slip, source_tag="ESP32_UART")
        self.assertTrue(success_slip)
        self.assertTrue(pkt_slip.is_valid)
        self.assertTrue(pkt_slip.slip_detected)
        self.assertEqual(pkt_slip.slip_intensity, 1.0)
        self.assertAlmostEqual(pkt_slip.total_force, 4.50, places=2)

    def test_20_packet_parser_malformed_telemetry(self):
        """Verify PacketParser safely catches malformed, corrupted, or out-of-range lines."""
        # A. Empty string
        pkt, ok = PacketParser.parse_raw_packet("", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)
        self.assertIn("Empty", pkt.parse_error)

        # B. Truncated fields (fewer than 8 comma-separated fields)
        pkt, ok = PacketParser.parse_raw_packet("3.85,0,1.54,0.85\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)
        self.assertIn("Malformed", pkt.parse_error)

        # C. Non-numeric strings
        pkt, ok = PacketParser.parse_raw_packet("3.85,ERR,1.54,0.85,0.77,0.42,0.27,245.0\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)

        # D. NaN or Inf values
        pkt, ok = PacketParser.parse_raw_packet("NaN,0,1.54,0.85,0.77,0.42,0.27,245.0\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)
        self.assertIn("NaN or Inf", pkt.parse_error)

        # E. Out-of-bounds total force (> 65.0N)
        pkt, ok = PacketParser.parse_raw_packet("75.50,0,1.54,0.85,0.77,0.42,0.27,245.0\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)
        self.assertIn("exceeds plausible range", pkt.parse_error)

        # F. Negative total force (< 0.0N)
        pkt, ok = PacketParser.parse_raw_packet("-4.50,0,1.54,0.85,0.77,0.42,0.27,245.0\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)

        # G. Invalid slip flag (not 0 or 1)
        pkt, ok = PacketParser.parse_raw_packet("3.85,9,1.54,0.85,0.77,0.42,0.27,245.0\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)
        self.assertIn("Slip flag must be 0 or 1", pkt.parse_error)

        # H. Out-of-bounds motor current (> 2500 mA)
        pkt, ok = PacketParser.parse_raw_packet("3.85,0,1.54,0.85,0.77,0.42,0.27,3200.0\n", source_tag="TEST")
        self.assertFalse(ok)
        self.assertFalse(pkt.is_valid)
        self.assertIn("Motor current", pkt.parse_error)

    def test_21_mock_serial_stream_and_fault_injection(self):
        """Verify MockSerialSensorInterface generates valid wire packets and injects faults."""
        mock = MockSerialSensorInterface()
        self.assertTrue(mock.is_connected())

        # 1. Normal packet generation
        pkt = mock.read_packet()
        self.assertTrue(pkt.is_valid)
        self.assertFalse(pkt.is_stale)
        self.assertIn("MOCK_SERIAL", pkt.source)

        # 2. Corrupt packet injection
        mock.inject_corrupt_packet()
        corrupt_pkt = mock.read_packet()
        self.assertFalse(corrupt_pkt.is_valid)
        self.assertGreater(mock.parse_errors_logged, 0)

        # Next packet automatically resumes normal valid stream
        resumed_pkt = mock.read_packet()
        self.assertTrue(resumed_pkt.is_valid)

        # 3. Stream stall injection
        mock.set_stall(True)
        stalled_pkt = mock.read_packet()
        self.assertTrue(stalled_pkt.is_stale)
        self.assertIn("stalled", stalled_pkt.parse_error.lower())
        mock.set_stall(False)

        # Resumed after un-stalling
        uninstalled_pkt = mock.read_packet()
        self.assertFalse(uninstalled_pkt.is_stale)

    def test_22_recorded_replay_telemetry(self):
        """Verify RecordedReplaySensorInterface deterministically steps through replay frames."""
        replay = RecordedReplaySensorInterface()
        self.assertTrue(replay.is_connected())
        self.assertFalse(replay.df.empty)

        initial_idx = replay.current_idx
        pkt1 = replay.read_packet()
        self.assertTrue(pkt1.is_valid)
        self.assertEqual(pkt1.source, "REPLAY")
        self.assertEqual(replay.current_idx, initial_idx + 1)

        # Step through multiple frames
        for _ in range(5):
            replay.read_packet()
        self.assertEqual(replay.current_idx, initial_idx + 6)

        # Reset replay
        replay.reset_replay()
        self.assertEqual(replay.current_idx, 0)

    def test_23_actuator_arming_and_emergency_stop(self):
        """Verify actuator safety gating: disarmed-by-default, software limits, emergency stop."""
        mock = MockSerialSensorInterface()

        # 1. Disarmed by default
        self.assertFalse(mock.is_armed())
        # Sending positive target force while disarmed is rejected
        rejected = mock.send_actuator_target(4.5)
        self.assertFalse(rejected)
        self.assertEqual(mock.target_force, 0.0)

        # 2. Arm actuators
        mock.arm_actuators()
        self.assertTrue(mock.is_armed())
        accepted = mock.send_actuator_target(4.5)
        self.assertTrue(accepted)
        self.assertEqual(mock.target_force, 4.5)

        # 3. Software safety clamp (cannot exceed max_software_limit_n)
        mock.send_actuator_target(50.0)
        self.assertEqual(mock.target_force, mock.max_software_limit_n)

        # 4. Software Emergency Stop
        mock.emergency_stop()
        self.assertTrue(mock.emergency_stopped)
        self.assertFalse(mock.is_armed())
        self.assertEqual(mock.target_force, 0.0)

        # While emergency stopped, arming is prohibited
        mock.arm_actuators()
        self.assertFalse(mock.is_armed())

        # Reset emergency stop
        mock.reset_emergency_stop()
        self.assertFalse(mock.emergency_stopped)
        self.assertFalse(mock.is_armed())  # Remains disarmed until explicit arm
        mock.arm_actuators()
        self.assertTrue(mock.is_armed())

    # ========================================================================
    # 8. ADAPTIVE CONTROLLER GRASP INFEASIBILITY & STATE MACHINE
    # ========================================================================

    def test_24_controller_grasp_infeasibility(self):
        """Verify that grasp infeasibility (F_slip_min > F_fragility) is detected and flagged."""
        controller = AdaptiveGripController()

        # Infeasible object: Extremely heavy, slippery, yet brittle
        # mass = 800g (0.8kg), mu = 0.08, fragility = 3.0N
        # F_min_slip = (0.8 * 9.80665) / (2 * 0.08) = 49.03 N >> 3.0 N
        infeasible_obj = ObjectSpec(
            name="InfeasibleGlassDrop",
            mass_g=800.0,
            friction_coeff=0.08,
            compliance_mm_per_n=0.1,
            fragility_limit_n=3.0
        )
        controller.configure(obj=infeasible_obj, base_target=2.5)
        self.assertTrue(controller.is_infeasible)
        self.assertIn("CRITICAL WARNING", controller.infeasibility_message)
        self.assertIn("physically infeasible", controller.infeasibility_message)

        # Feasible standard object: Cup
        cup_obj = STANDARD_OBJECTS["Cup"]
        controller.configure(obj=cup_obj, base_target=3.5)
        self.assertFalse(controller.is_infeasible)
        self.assertEqual(controller.infeasibility_message, "")

    def test_25_controller_state_machine_legal_transitions_and_watchdog(self):
        """Verify state machine legal transitions and watchdog response to stale packets."""
        controller = AdaptiveGripController()
        sensor = SimulatedSensorInterface()
        controller.configure(STANDARD_OBJECTS["Cup"], base_target=3.85)

        # Initial state is OPEN
        self.assertEqual(controller.state, HandState.OPEN)

        # Illegal transition: OPEN -> SLIPPING (not permitted)
        legal = controller._transition_to(HandState.SLIPPING)
        self.assertFalse(legal)
        self.assertEqual(controller.state, HandState.OPEN)

        # Legal transition: OPEN -> CLOSING
        legal = controller._transition_to(HandState.CLOSING)
        self.assertTrue(legal)
        self.assertEqual(controller.state, HandState.CLOSING)

        # Legal transition: CLOSING -> GRASPING
        legal = controller._transition_to(HandState.GRASPING)
        self.assertTrue(legal)
        self.assertEqual(controller.state, HandState.GRASPING)

        # Watchdog: feed stale packets into controller step
        stale_pkt = SensorPacket(
            timestamp=0.0,
            source="MOCK",
            is_valid=False,
            is_stale=True,
            parse_error="Watchdog test stale packet"
        )
        # Mock sensor returning stale packet
        class StaleSensor(SimulatedSensorInterface):
            def read_packet(self):
                return stale_pkt

        stale_sensor = StaleSensor()
        for i in range(5):
            controller.step(stale_sensor)

        # Verify controller responded to stale packet stream
        self.assertGreaterEqual(controller.stale_packet_count, 5)
        self.assertEqual(controller.state, HandState.OPEN)
        self.assertIn("WATCHDOG TIMEOUT", controller.last_action_note)

    # ========================================================================
    # 9. PERSONALIZATION CRUD, FILTER, BACKUP, RESTORE & JSON IMPORT/EXPORT
    # ========================================================================

    def test_26_personalization_crud_filter_backup_restore_and_json_import(self):
        """Verify profile CRUD, filtering, JSON import/export, and backup snapshots."""
        pm = PersonalizationManager()

        # 1. Profile CRUD
        success, msg = pm.create_profile("UnitTester", "Firm", 1.70)
        self.assertTrue(success)
        self.assertIn("UnitTester", pm.profiles)
        self.assertEqual(pm.profiles["UnitTester"].preset_mode, "Firm")
        self.assertEqual(pm.profiles["UnitTester"].base_safety_margin, 1.70)

        # Duplicate rejection
        dup_success, _ = pm.create_profile("UnitTester")
        self.assertFalse(dup_success)

        # Delete profile
        del_success, _ = pm.delete_profile("UnitTester")
        self.assertTrue(del_success)
        self.assertNotIn("UnitTester", pm.profiles)

        # 2. Filter trials
        filtered = pm.filter_trials(object_name="Plastic Bottle")
        for t in filtered:
            self.assertEqual(t.object_name, "Plastic Bottle")

        # 3. Export JSON
        exported_json = pm.export_trials_json()
        self.assertIsInstance(exported_json, str)
        self.assertIn("Plastic Bottle", exported_json)

        # 4. Import JSON with schema validation
        valid_json = """[
            {
                "trial_id": "TEST-IMP-01",
                "timestamp": "2026-10-29 12:00:00",
                "object_name": "Test Foam Ball",
                "user_profile": "Alex (Standard)",
                "preset_mode": "Standard",
                "initial_target_f": 2.10,
                "final_target_f": 2.30,
                "peak_measured_f": 2.35,
                "slip_events": 1,
                "outcome": "ADAPTED_SUCCESS",
                "feedback_rating": "Ideal",
                "notes": "Automated import test entry"
            }
        ]"""
        n_imported, errs = pm.import_trials_json(valid_json)
        self.assertEqual(n_imported, 1)
        self.assertEqual(len(errs), 0)
        self.assertTrue(pm.trials[0].trial_id.startswith("TEST-IMP-01"))

        # Malformed JSON import test
        bad_json = "[{'broken': true}]"
        n_imported_bad, bad_errs = pm.import_trials_json(bad_json)
        self.assertEqual(n_imported_bad, 0)
        self.assertGreater(len(bad_errs), 0)

        # 5. Backup & Restore
        backup_file = pm.create_backup()
        self.assertTrue(backup_file.endswith(".json"))
        backups = pm.list_backups()
        backup_names = [b["filename"] for b in backups]
        self.assertIn(backup_file, backup_names)

        restore_ok, r_msg = pm.restore_backup(backup_file)
        self.assertTrue(restore_ok)
        self.assertIn("Successfully restored", r_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
