import isaacgym
from isaacgym import gymapi

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

sim_params.physx.use_gpu = False
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(
    0,
    0,
    gymapi.SIM_PHYSX,
    sim_params
)

if sim is None:
    raise RuntimeError("Simulation creation failed")

print("Simulation created successfully")

viewer = gym.create_viewer(
    sim,
    gymapi.CameraProperties()
)

if viewer is None:
    raise RuntimeError("Viewer creation failed")

print("Viewer created successfully")

while not gym.query_viewer_has_closed(viewer):
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

print("Viewer closed")
