import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define all cubes
cubes = [
    {"id":0, "x":0, "y":0, "z":0, "w":7, "h":7, "l":7},
    {"id":1, "x":0, "y":7, "z":0, "w":7, "h":7, "l":7},
    {"id":2, "x":1, "y":0, "z":8, "w":7, "h":7, "l":7},
    {"id":3, "x":1, "y":7, "z":7, "w":7, "h":7, "l":7},
    {"id":4, "x":8, "y":0, "z":0, "w":7, "h":7, "l":7},
    {"id":5, "x":8, "y":0, "z":8, "w":7, "h":7, "l":7},
    {"id":6, "x":8, "y":7, "z":0, "w":7, "h":7, "l":7},
    {"id":7, "x":0, "y":7, "z":14, "w":7, "h":7, "l":6},
    {"id":8, "x":0, "y":14, "z":0, "w":7, "h":6, "l":7},
    {"id":9, "x":8, "y":7, "z":7, "w":7, "h":6, "l":7},
    {"id":10,"x":0, "y":14, "z":7, "w":7, "h":6, "l":6},
    {"id":11,"x":7, "y":14, "z":0, "w":7, "h":5, "l":7},
    {"id":12,"x":8, "y":0, "z":15,"w":7, "h":7, "l":5},
    {"id":13,"x":15,"y":0, "z":13,"w":5, "h":7, "l":7},
    {"id":14,"x":15,"y":7, "z":1, "w":5, "h":7, "l":7},
    {"id":15,"x":14,"y":14,"z":0, "w":6, "h":6, "l":6},
    {"id":16,"x":7, "y":7, "z":14,"w":7, "h":5, "l":6},
    {"id":17,"x":7, "y":14,"z":7, "w":7, "h":5, "l":6},
    {"id":18,"x":0, "y":14,"z":13,"w":5, "h":5, "l":7},
    {"id":19,"x":7, "y":14,"z":13,"w":5, "h":4, "l":7},
]

# Intersection check
def intersects(a, b):
    return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and
            a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"] and
            a["z"] < b["z"] + b["l"] and b["z"] < a["z"] + a["l"])

print("Intersecting cubes:")
for i in range(len(cubes)):
    for j in range(i+1, len(cubes)):
        if intersects(cubes[i], cubes[j]):
            print(f"Cube {cubes[i]['id']} intersects Cube {cubes[j]['id']}")

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab20.colors  # 20 distinct colors

def draw_cube(ax, cube, color):
    x, y, z, w, h, l = cube["x"], cube["y"], cube["z"], cube["w"], cube["h"], cube["l"]
    # 8 vertices
    v = [
        [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
        [x, y, z+l], [x+w, y, z+l], [x+w, y+h, z+l], [x, y+h, z+l]
    ]
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolor=color, edgecolor="black"))

for i, cube in enumerate(cubes):
    draw_cube(ax, cube, colors[i % len(colors)])

# Adjust limits automatically
max_x = max(c["x"] + c["w"] for c in cubes)
max_y = max(c["y"] + c["h"] for c in cubes)
max_z = max(c["z"] + c["l"] for c in cubes)

ax.set_xlim(0, max_x+1)
ax.set_ylim(0, max_y+1)
ax.set_zlim(0, max_z+1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
