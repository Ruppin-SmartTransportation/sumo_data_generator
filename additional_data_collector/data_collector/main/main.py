import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.simulation_runner import SimulationRunner

if __name__ == "__main__":
    delay = 0
    num_of_steps = 100

    simulations = [
        ("holon_morning.sumocfg", "morning"),
        ("holon_noon.sumocfg", "noon"),
        ("holon_evening.sumocfg", "evening"),
    ]

    for cfg, tag in simulations:
        print(f"\n🚦 Running {cfg} simulation...\n")
        sim = SimulationRunner(delay, num_of_steps, cfg, tag)
        sim.run_simulation()
