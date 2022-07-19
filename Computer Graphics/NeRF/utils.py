""" 
Some utility functions
"""
import open3d as o3d
import numpy as np

# Pytorch 1.9
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

# Pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)


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


def create_pytorch3d_renderer(device, radius=3.7, elevation=0, azimuth=180, fov=40.0, width=512, lights=None):
    # render the tri-mesh using pytorch3d
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    R, T = look_at_view_transform(radius, elevation, azimuth) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=width, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    if lights is None:
        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, -1e5]], ambient_color=[[0, 0, 0]],
                         specular_color=[[0., 0., 0.]], diffuse_color=[[1., 1., 1.]])
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    return renderer


def pytorch3d_mesh(device, V, T, F):
    shape = torch.from_numpy(V).to(device)[None]
    color = TexturesVertex(torch.from_numpy(T).to(device)[None])
    tri = torch.from_numpy(F).to(device)[None]
    mesh = Meshes(shape, tri.repeat(1, 1, 1), color)
    return mesh


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
    return np.array(V, dtype=np.float32), np.array(T, dtype=np.float32), np.array(F, dtype=np.float32)-1
    
    
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

            

