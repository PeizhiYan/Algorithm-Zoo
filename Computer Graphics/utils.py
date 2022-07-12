""" 
Some utility functions
"""
import open3d as o3d
import numpy as np
import pytorch3d

def o3d_form_mesh(V, T, F, smoothing=False):
    """
    form an open3d tri-mesh object
     V: vertices
     T: colors
     F: triangle faces
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V) # dtype vector3d (float)
    mesh.triangles = o3d.utility.Vector3iVector(F) # dtype vector3i (int)
    if smoothing:
        # smooth the mesh using Laplacian filter
        mesh = mesh.filter_smooth_laplacian(1, 0.5,
                filter_scope=o3d.geometry.FilterScope.Vertex)
    if T is not None:
        if len(T) > 0:
            mesh.vertex_colors = o3d.utility.Vector3dVector(T) # dtype vector3i (int)
    mesh.compute_vertex_normals() # computing normal will give specular effect while rendering
    return mesh


def o3d_render(mesh, width=512, height=512, normal=False):
    # render the tri-mesh using open3d
    mesh.compute_triangle_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible = False)
    opt = vis.get_render_option()
    if normal:
        # use surface normal as texture
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
    vis.add_geometry(mesh)
    # smooth shading
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    depth = vis.capture_depth_float_buffer(True)
    image = vis.capture_screen_float_buffer(True)
    return np.asarray(image), np.asarray(depth)


def load_obj(path):
    """
    Load mesh from .obj file
    """
    V = []
    T = []
    F = [] 
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.split(' ')
            if arr[0] in ['v', 'V', 'f', 'F', 't', 'T']:
                x = []
                for i in range(3):
                    x.append(float(arr[i+1]))
                if arr[0] in ['v', 'V']:
                    V.append(x)
                if arr[0] in ['f', 'F']:
                    F.append(x)
                if arr[0] in ['t', 'T']:
                    T.append(x)
    return np.array(V), np.array(T), np.array(F)-1
    
    
def save_mesh_as_obj(mesh, path):
    """ 
    Save Open3D mesh to a given path with format [int].obj
    """
    mesh.compute_vertex_normals() # compute vertex normals
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    VN = np.asarray(mesh.vertex_normals) 
    with open(path, 'w') as f:
        f.write("# OBJ file\n")
        f.write("# Vertices: {}\n".format(len(V)))
        f.write("# Faces: {}\n".format(len(F)))
        for vid in range(len(V)):
            v = V[vid]
            vn = VN[vid]
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            f.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
        for p in F:
            f.write("f")
            for i in p:
                f.write(" {}".format((i + 1)))
            f.write("\n")

            

