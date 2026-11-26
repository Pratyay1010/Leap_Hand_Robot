import isaacgym
from isaacgym import gymapi
import torch

print("=" * 50)
print("Isaac Gym Diagnostic")
print("=" * 50)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

print("\nImporting Isaac Gym...")
gym = gymapi.acquire_gym()

print("Gym acquired successfully.")

sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = False

print("\nCreating PhysX simulation...")

sim = gym.create_sim(
    0,
    0,
    gymapi.SIM_PHYSX,
    sim_params
)

print("Simulation object:", sim)

if sim is None:

    print("\nFAILED: Isaac Gym simulation creation returned None.")

else:

    print("\nSUCCESS: Isaac Gym simulation created.")

    print("Destroying simulation...")

    gym.destroy_sim(sim)

    print("SUCCESS: Simulation destroyed.")

print("\nDiagnostic completed.")