import airsim

def draw_target(client, pos, color=[1,1,0,1]):
    client.simPlotPoints(
        [airsim.Vector3r(*pos)],
        color, size=20, duration=60, is_persistent=True
    )

def draw_velocity(client, start, vec, color=[1,0,0,1]):
    end = start + vec
    client.simDrawLine(
        airsim.Vector3r(*start),
        airsim.Vector3r(*end),
        color, thickness=3, duration=0.2
    )

def print_state(client, drone, state):
    client.simPrintText(
        f"{drone}: {state.name}",
        airsim.Vector3r(0,0,-2),
        size=15
    )
