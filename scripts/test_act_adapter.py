
import mujoco
import numpy as np

from rlact.envs.aloha_transfer import AlohaTransferTask
from rlact.policies.act.adapter import ACTAdapter


task = AlohaTransferTask()
renderer = None

try:
    task.reset(seed=42)

    renderer = mujoco.Renderer(
        task.model,
        height=480,
        width=640,
    )

    renderer.update_scene(task.data, camera="top")
    image = renderer.render().copy()

    qpos = task.data.qpos

    # Convert MuJoCo joint positions to ACT's 14D state
    state = np.concatenate([
        qpos[0:6],
        [(qpos[6] - 0.01844) / (0.05800 - 0.01844)],
        qpos[8:14],
        [(qpos[14] - 0.01844) / (0.05800 - 0.01844)],
    ])

    print("Image shape:", image.shape)
    print("State shape:", state.shape)

    adapter = ACTAdapter()
    adapter.reset()

    actions = adapter.predict_chunk(image, state)

    print("Predicted action shape:", actions.shape)
    print("All actions finite:", np.isfinite(actions).all())
    print("First action:", np.round(actions[0], 3))

finally:
    if renderer is not None:
        renderer.close()
    task.close()
