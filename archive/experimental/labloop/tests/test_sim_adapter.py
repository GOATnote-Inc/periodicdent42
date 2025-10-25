import threading

from labloop.orchestrator.adapters.sim import SimDevice
from labloop.orchestrator.models.schemas import InstrumentType, Task, TaskType


def test_sim_device_generates_points():
    device = SimDevice(instrument=InstrumentType.XRD, seed=1)
    task = Task(
        id="task",
        type=TaskType.XRD,
        parameters={
            "two_theta_start": 20,
            "two_theta_end": 25,
            "step": 1,
            "dwell_time": 0.001,
        },
    )
    measurement = device.execute(task, threading.Event())
    assert measurement.data["type"] == "xrd"
    assert len(measurement.data["points"]) > 0
