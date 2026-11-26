import trimesh

mesh = trimesh.load("model.stl")

print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.faces))
print("Dimensions:", mesh.bounds)
print("Size:", mesh.extents)
print("Watertight:", mesh.is_watertight)