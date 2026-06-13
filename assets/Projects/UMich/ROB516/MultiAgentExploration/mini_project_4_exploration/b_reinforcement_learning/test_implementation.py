"""
Quick test script to verify the RL implementation works.
Run this to check that all components are functional.
"""

import numpy as np
from b_reinforcement_learning import (
    CurriculumMapGenerator,
    RewardCalculator,
    ExplorationEnv,
)

def test_curriculum():
    """Test curriculum map generation."""
    print("Testing curriculum map generation...")
    gen = CurriculumMapGenerator(size=20, seed=42)
    
    for level in range(3):
        truth, robot_start = gen.generate(level)
        assert truth.shape == (20, 20), f"Level {level}: Wrong shape"
        assert 0 <= robot_start[0] < 20 and 0 <= robot_start[1] < 20, f"Level {level}: Invalid start"
        free_cells = np.sum(truth == 0)
        assert free_cells > 0, f"Level {level}: No free cells"
        print(f"  ✓ Level {level}: {free_cells} free cells, start={robot_start}")
    
    print("✓ Curriculum test passed!\n")

def test_rewards():
    """Test reward calculations."""
    print("Testing reward calculations...")
    calc = RewardCalculator(alpha=1.0, beta=5.0, lambda_=0.5)
    
    # Test extrinsic
    r_ext = calc.calculate_extrinsic_reward(3)
    assert r_ext == 3.0, f"Extrinsic reward wrong: {r_ext}"
    
    # Test safety
    r_safety_collision = calc.calculate_safety_penalty(True)
    r_safety_no_collision = calc.calculate_safety_penalty(False)
    assert r_safety_collision == -5.0, f"Safety penalty wrong: {r_safety_collision}"
    assert r_safety_no_collision == 0.0, f"Safety (no collision) wrong: {r_safety_no_collision}"
    
    # Test intrinsic
    r_int = calc.calculate_intrinsic_reward(0.8)
    assert abs(r_int - (-0.4)) < 1e-6, f"Intrinsic reward wrong: {r_int}"
    
    # Test total
    r_total, r_dict = calc.calculate_total_reward(3, False, 0.2)
    expected = 3.0 + 0.0 + (-0.1)
    assert abs(r_total - expected) < 1e-6, f"Total reward wrong: {r_total}"
    
    print("  ✓ Extrinsic reward: OK")
    print("  ✓ Safety penalty: OK")
    print("  ✓ Intrinsic reward: OK")
    print("  ✓ Total reward: OK")
    print("✓ Reward test passed!\n")

def test_environment():
    """Test Gymnasium environment."""
    print("Testing Gymnasium environment...")
    env = ExplorationEnv(size=20, sensor_radius=5, seed=42)
    
    # Test reset
    obs, info = env.reset(options={'level': 0})
    assert obs.shape == (3, 11, 11), f"Observation shape wrong: {obs.shape}"
    assert obs.dtype == np.float32, f"Observation dtype wrong: {obs.dtype}"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, f"Observation range wrong: [{obs.min()}, {obs.max()}]"
    print(f"  ✓ Reset: obs shape={obs.shape}, dtype={obs.dtype}")
    
    # Test step
    action = 0  # UP
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (3, 11, 11), f"Step observation shape wrong: {obs.shape}"
    assert isinstance(reward, float), f"Reward type wrong: {type(reward)}"
    assert isinstance(info['coverage'], float), f"Coverage type wrong: {type(info['coverage'])}"
    print(f"  ✓ Step: reward={reward:.2f}, coverage={info['coverage']:.2%}")
    
    # Test multiple steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"  ✓ Multiple steps: coverage={info['coverage']:.2%}")
    print("✓ Environment test passed!\n")

def test_integration():
    """Test end-to-end integration."""
    print("Testing integration...")
    
    env = ExplorationEnv(size=20, sensor_radius=5, seed=42)
    obs, info = env.reset(options={'level': 0})
    
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"  ✓ Random policy: {step+1} steps, coverage={info['coverage']:.2%}, reward={total_reward:.2f}")
    
    # Verify that coverage increased
    assert info['coverage'] > 0.1, "Coverage should increase with random exploration"
    
    print("✓ Integration test passed!\n")

if __name__ == "__main__":
    print("="*60)
    print("Part B: RL Implementation Test Suite")
    print("="*60)
    print()
    
    try:
        test_curriculum()
        test_rewards()
        test_environment()
        test_integration()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe RL implementation is working correctly.")
        print("You can now proceed with training in the notebook.")
        
    except AssertionError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nPlease check your implementation.")
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR DURING TESTING")
        print("="*60)
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()
