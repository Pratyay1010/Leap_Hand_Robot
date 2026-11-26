import os
import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil


# ============================================================
# CONFIGURATION
# ============================================================

ASSET_ROOT = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = "robot.urdf"

STEP_SIZE = 0.05  # radians per keyboard press


# ============================================================
# INITIALIZE ISAAC GYM
# ============================================================

gym = gymapi.acquire_gym()

args = gymutil.parse_arguments()

sim_params = gymapi.SimParams()

sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 2

sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    gymapi.SIM_PHYSX,
    sim_params
)

if sim is None:
    raise RuntimeError("Failed to create Isaac Gym simulation")


# ============================================================
# GROUND
# ============================================================

plane_params = gymapi.PlaneParams()

plane_params.normal = gymapi.Vec3(0, 0, 1)

gym.add_ground(sim, plane_params)


# ============================================================
# LOAD ROBOT
# ============================================================

asset_options = gymapi.AssetOptions()

asset_options.fix_base_link = True

asset_options.disable_gravity = False

asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset_options.use_mesh_materials = True

asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

asset_options.collapse_fixed_joints = False

asset_options.armature = 0.01

asset_options.override_com = False

asset_options.override_inertia = False

asset_options.vhacd_enabled = False


robot_asset = gym.load_asset(
    sim,
    ASSET_ROOT,
    URDF_FILE,
    asset_options
)

if robot_asset is None:
    raise RuntimeError("Failed to load robot URDF")


# ============================================================
# PRINT ROBOT INFORMATION
# ============================================================

num_dofs = gym.get_asset_dof_count(robot_asset)

num_links = gym.get_asset_rigid_body_count(robot_asset)

num_joints = gym.get_asset_joint_count(robot_asset)

dof_names = gym.get_asset_dof_names(robot_asset)

joint_names = gym.get_asset_joint_names(robot_asset)

print("\n" + "=" * 60)
print("ROBOT INFORMATION")
print("=" * 60)

print("Number of links:", num_links)
print("Number of joints:", num_joints)
print("Number of DOFs:", num_dofs)

print("\nDOF NAMES:")

for i, name in enumerate(dof_names):
    print(f"{i}: {name}")

print("\nJOINT NAMES:")

for i, name in enumerate(joint_names):
    print(f"{i}: {name}")

print("=" * 60)


# ============================================================
# ENVIRONMENT
# ============================================================

env = gym.create_env(
    sim,
    gymapi.Vec3(-1.0, -1.0, 0.0),
    gymapi.Vec3(1.0, 1.0, 1.0),
    1
)


# ============================================================
# ROBOT INITIAL POSITION
# ============================================================

robot_pose = gymapi.Transform()

robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)

robot_pose.r = gymapi.Quat.from_euler_zyx(
    0.0,
    0.0,
    0.0
)


robot = gym.create_actor(
    env,
    robot_asset,
    robot_pose,
    "robot_hand",
    0,
    1
)


# ============================================================
# CONFIGURE JOINTS
# ============================================================

dof_properties = gym.get_actor_dof_properties(env, robot)

dof_properties["driveMode"].fill(gymapi.DOF_MODE_POS)

dof_properties["stiffness"].fill(100.0)

dof_properties["damping"].fill(10.0)

gym.set_actor_dof_properties(
    env,
    robot,
    dof_properties
)


# ============================================================
# JOINT LIMITS
# ============================================================

lower_limits = dof_properties["lower"]

upper_limits = dof_properties["upper"]

print("\nJOINT LIMITS:")

for i in range(num_dofs):

    print(
        f"{i}: {dof_names[i]} | "
        f"Lower: {lower_limits[i]:.3f} | "
        f"Upper: {upper_limits[i]:.3f}"
    )


# ============================================================
# INITIAL DOF STATES
# ============================================================

initial_dof_states = gym.get_actor_dof_states(
    env,
    robot,
    gymapi.STATE_ALL
)

initial_positions = initial_dof_states["pos"].copy()

target_positions = initial_positions.copy()


# ============================================================
# SIMULATION PREPARATION
# ============================================================

gym.prepare_sim(sim)


# ============================================================
# CREATE VIEWER
# ============================================================

viewer = gym.create_viewer(
    sim,
    gymapi.CameraProperties()
)

if viewer is None:
    raise RuntimeError("Failed to create viewer")


gym.viewer_camera_look_at(
    viewer,
    env,
    gymapi.Vec3(0.5, 0.5, 0.5),
    gymapi.Vec3(0.0, 0.0, 0.5)
)


# ============================================================
# KEYBOARD CONTROLS
# ============================================================

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_LEFT,
    "LEFT"
)

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_RIGHT,
    "RIGHT"
)

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_UP,
    "UP"
)

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_DOWN,
    "DOWN"
)

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_R,
    "RESET"
)

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_SPACE,
    "TOGGLE"
)

gym.subscribe_viewer_keyboard_event(
    viewer,
    gymapi.KEY_Q,
    "QUIT"
)


# ============================================================
# INTERACTIVE CONTROL
# ============================================================

selected_dof = 0

simulation_running = True

print("\n" + "=" * 60)
print("CONTROLS")
print("=" * 60)

print("LEFT / RIGHT : Select joint")
print("UP           : Increase selected joint angle")
print("DOWN         : Decrease selected joint angle")
print("R            : Reset robot")
print("SPACE        : Pause / Resume simulation")
print("Q            : Quit")

print("=" * 60)


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

while not gym.query_viewer_has_closed(viewer):

    # --------------------------------------------------------
    # KEYBOARD EVENTS
    # --------------------------------------------------------

    events = gym.query_viewer_action_events(viewer)

    for event in events:

        if event.value != 1.0:
            continue

        if event.action == "QUIT":
            break

        elif event.action == "LEFT":

            selected_dof = max(
                0,
                selected_dof - 1
            )

            print(
                f"\nSelected DOF: "
                f"{selected_dof} - "
                f"{dof_names[selected_dof]}"
            )

        elif event.action == "RIGHT":

            selected_dof = min(
                num_dofs - 1,
                selected_dof + 1
            )

            print(
                f"\nSelected DOF: "
                f"{selected_dof} - "
                f"{dof_names[selected_dof]}"
            )

        elif event.action == "UP":

            target_positions[selected_dof] += STEP_SIZE

            target_positions[selected_dof] = np.clip(
                target_positions[selected_dof],
                lower_limits[selected_dof],
                upper_limits[selected_dof]
            )

            print(
                f"{dof_names[selected_dof]} = "
                f"{target_positions[selected_dof]:.3f} rad"
            )

        elif event.action == "DOWN":

            target_positions[selected_dof] -= STEP_SIZE

            target_positions[selected_dof] = np.clip(
                target_positions[selected_dof],
                lower_limits[selected_dof],
                upper_limits[selected_dof]
            )

            print(
                f"{dof_names[selected_dof]} = "
                f"{target_positions[selected_dof]:.3f} rad"
            )

        elif event.action == "RESET":

            target_positions = initial_positions.copy()

            gym.set_actor_dof_states(
                env,
                robot,
                initial_dof_states,
                gymapi.STATE_ALL
            )

            print("Robot reset")

        elif event.action == "TOGGLE":

            simulation_running = not simulation_running

            print(
                "Simulation:",
                "RUNNING" if simulation_running else "PAUSED"
            )


    # --------------------------------------------------------
    # APPLY JOINT TARGETS
    # --------------------------------------------------------

    gym.set_actor_dof_position_targets(
        env,
        robot,
        target_positions
    )


    # --------------------------------------------------------
    # SIMULATE
    # --------------------------------------------------------

    if simulation_running:

        gym.simulate(sim)

        gym.fetch_results(sim, True)


    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    gym.step_graphics(sim)

    gym.draw_viewer(
        viewer,
        sim,
        True
    )

    gym.sync_frame_time(sim)


# ============================================================
# CLEANUP
# ============================================================

gym.destroy_viewer(viewer)

gym.destroy_sim(sim)

print("Simulation closed")