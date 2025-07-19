// module for processing .glb format files compliant with the glTF 2.0 standard
// Animation files are calculated using Rest Pose.

use alloc::vec::Vec;
use alloc::string::String;
use gltf::Gltf; 
use gltf::accessor::{Accessor,DataType, Dimensions};
use gltf::mesh::{Semantic, Mode};
use core::convert::TryInto; 

use cgmath::{Matrix4, Point3, Vector3, InnerSpace, Transform as CgmathTransform, SquareMatrix, Matrix, Zero};

#[derive(Debug, Clone)]
pub struct GltfMesh {
    pub name: Option<String>,
    pub vertices: Vec<[f32; 3]>, 
    pub normals: Option<Vec<[f32; 3]>>, 
    pub indices: Option<Vec<u32>>,     
}

#[derive(Debug)]
pub enum GltfError {
    GltfImportError(gltf::Error),
    IoError, 
    PrimitiveNotTriangles,
    UnsupportedAccessorFormat,
    MissingPositions,
    IndexOutOfBounds,
    BufferReadError,
    MissingVertexNormals,
    SkinningError, 
}

impl From<gltf::Error> for GltfError {
    fn from(err: gltf::Error) -> Self {
        GltfError::GltfImportError(err)
    }
}

fn read_f32_vec3_data(accessor: &Accessor, blob_data: Option<&[u8]>) -> Result<Vec<[f32; 3]>, GltfError> {
    let view = accessor.view().ok_or(GltfError::UnsupportedAccessorFormat)?;
    let buffer = view.buffer();
    if buffer.index() != 0 { return Err(GltfError::BufferReadError); } 

    let data_slice = blob_data.ok_or(GltfError::BufferReadError)?;
    let total_offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or_else(|| accessor.size()); 

    if accessor.data_type() != DataType::F32 || accessor.dimensions() != Dimensions::Vec3 {
        return Err(GltfError::UnsupportedAccessorFormat);
    }

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let item_offset = total_offset + i * stride;
        if item_offset + 12 > data_slice.len() { 
             return Err(GltfError::IndexOutOfBounds); 
        }
        let x = f32::from_le_bytes(data_slice[item_offset..item_offset+4].try_into().map_err(|_| GltfError::BufferReadError)?);
        let y = f32::from_le_bytes(data_slice[item_offset+4..item_offset+8].try_into().map_err(|_| GltfError::BufferReadError)?);
        let z = f32::from_le_bytes(data_slice[item_offset+8..item_offset+12].try_into().map_err(|_| GltfError::BufferReadError)?);
        result.push([x, y, z]);
    }
    Ok(result)
}

fn read_f32_vec4_data(accessor: &Accessor, blob_data: Option<&[u8]>) -> Result<Vec<[f32; 4]>, GltfError> {
    let view = accessor.view().ok_or(GltfError::UnsupportedAccessorFormat)?;
    if view.buffer().index() != 0 { return Err(GltfError::BufferReadError); }
    let data_slice = blob_data.ok_or(GltfError::BufferReadError)?;
    let total_offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or_else(|| accessor.size());
    if accessor.data_type() != DataType::F32 || accessor.dimensions() != Dimensions::Vec4 { return Err(GltfError::UnsupportedAccessorFormat); }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let item_offset = total_offset + i * stride;
        if item_offset + 16 > data_slice.len() { return Err(GltfError::IndexOutOfBounds); }
        let x = f32::from_le_bytes(data_slice[item_offset..item_offset+4].try_into().map_err(|_| GltfError::BufferReadError)?);
        let y = f32::from_le_bytes(data_slice[item_offset+4..item_offset+8].try_into().map_err(|_| GltfError::BufferReadError)?);
        let z = f32::from_le_bytes(data_slice[item_offset+8..item_offset+12].try_into().map_err(|_| GltfError::BufferReadError)?);
        let w = f32::from_le_bytes(data_slice[item_offset+12..item_offset+16].try_into().map_err(|_| GltfError::BufferReadError)?);
        result.push([x, y, z, w]);
    }
    Ok(result)
}

fn read_u8_vec4_data(accessor: &Accessor, blob_data: Option<&[u8]>) -> Result<Vec<[u8; 4]>, GltfError> {
    let view = accessor.view().ok_or(GltfError::UnsupportedAccessorFormat)?;
    if view.buffer().index() != 0 { return Err(GltfError::BufferReadError); }
    let data_slice = blob_data.ok_or(GltfError::BufferReadError)?;
    let total_offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or_else(|| accessor.size());
    if accessor.data_type() != DataType::U8 || accessor.dimensions() != Dimensions::Vec4 { return Err(GltfError::UnsupportedAccessorFormat); }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let item_offset = total_offset + i * stride;
        if item_offset + 4 > data_slice.len() { return Err(GltfError::IndexOutOfBounds); }
        let x = data_slice[item_offset];
        let y = data_slice[item_offset+1];
        let z = data_slice[item_offset+2];
        let w = data_slice[item_offset+3];
        result.push([x, y, z, w]);
    }
    Ok(result)
}

fn read_u16_vec4_data(accessor: &Accessor, blob_data: Option<&[u8]>) -> Result<Vec<[u16; 4]>, GltfError> {
    let view = accessor.view().ok_or(GltfError::UnsupportedAccessorFormat)?;
    if view.buffer().index() != 0 { return Err(GltfError::BufferReadError); }
    let data_slice = blob_data.ok_or(GltfError::BufferReadError)?;
    let total_offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or_else(|| accessor.size());
    if accessor.data_type() != DataType::U16 || accessor.dimensions() != Dimensions::Vec4 { return Err(GltfError::UnsupportedAccessorFormat); }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let item_offset = total_offset + i * stride;
        if item_offset + 8 > data_slice.len() { return Err(GltfError::IndexOutOfBounds); }
        let x = u16::from_le_bytes(data_slice[item_offset..item_offset+2].try_into().map_err(|_| GltfError::BufferReadError)?);
        let y = u16::from_le_bytes(data_slice[item_offset+2..item_offset+4].try_into().map_err(|_| GltfError::BufferReadError)?);
        let z = u16::from_le_bytes(data_slice[item_offset+4..item_offset+6].try_into().map_err(|_| GltfError::BufferReadError)?);
        let w = u16::from_le_bytes(data_slice[item_offset+6..item_offset+8].try_into().map_err(|_| GltfError::BufferReadError)?);
        result.push([x, y, z, w]);
    }
    Ok(result)
}

fn read_mat4_data(accessor: &Accessor, blob_data: Option<&[u8]>) -> Result<Vec<Matrix4<f32>>, GltfError> {
    let view = accessor.view().ok_or(GltfError::UnsupportedAccessorFormat)?;
    if view.buffer().index() != 0 { return Err(GltfError::BufferReadError); }
    let data_slice = blob_data.ok_or(GltfError::BufferReadError)?;
    let total_offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or_else(|| accessor.size());
    if accessor.data_type() != DataType::F32 || accessor.dimensions() != Dimensions::Mat4 { return Err(GltfError::UnsupportedAccessorFormat); }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let item_offset = total_offset + i * stride;
        if item_offset + 64 > data_slice.len() { return Err(GltfError::IndexOutOfBounds); }
        let mut mat_data = [0.0f32; 16];
        for j in 0..16 {
            mat_data[j] = f32::from_le_bytes(data_slice[item_offset + j*4 .. item_offset + (j+1)*4].try_into().map_err(|_| GltfError::BufferReadError)?);
        }
        result.push(Matrix4::new(
            mat_data[0], mat_data[1], mat_data[2], mat_data[3],
            mat_data[4], mat_data[5], mat_data[6], mat_data[7],
            mat_data[8], mat_data[9], mat_data[10], mat_data[11],
            mat_data[12], mat_data[13], mat_data[14], mat_data[15],
        ));
    }
    Ok(result)
}


fn read_indices_data(accessor: &Accessor, blob_data: Option<&[u8]>) -> Result<Vec<u32>, GltfError> {
    let view = accessor.view().ok_or(GltfError::UnsupportedAccessorFormat)?;
    let buffer = view.buffer();
    if buffer.index() != 0 { return Err(GltfError::BufferReadError); }
    let data_slice = blob_data.ok_or(GltfError::BufferReadError)?;
    let total_offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or_else(|| accessor.size()); 
    if accessor.dimensions() != Dimensions::Scalar { return Err(GltfError::UnsupportedAccessorFormat); }
    let mut result = Vec::with_capacity(count);
    match accessor.data_type() {
        DataType::U8 => { for i in 0..count { let offset = total_offset + i * stride; result.push(data_slice[offset] as u32); } }
        DataType::U16 => { for i in 0..count { let offset = total_offset + i * stride; result.push(u16::from_le_bytes(data_slice[offset..offset+2].try_into().unwrap()) as u32); } }
        DataType::U32 => { for i in 0..count { let offset = total_offset + i * stride; result.push(u32::from_le_bytes(data_slice[offset..offset+4].try_into().unwrap())); } }
        _ => return Err(GltfError::UnsupportedAccessorFormat),
    }
    Ok(result)
}

fn process_mesh_data(
    mesh: gltf::Mesh, 
    node_transform_cg: Matrix4<f32>, 
    skin: Option<&gltf::Skin>,
    blob_data: Option<&[u8]>,
    joint_global_transforms: &[Matrix4<f32>],
) -> Result<GltfMesh, GltfError> {
    let mut aggregated_vertices = Vec::new();
    let mut aggregated_normals_vec: Vec<[f32; 3]> = Vec::new();
    let mut aggregated_indices_vec: Vec<u32> = Vec::new();
    let mut current_vertex_offset = 0;
    let mut has_any_normals = false;

    let skin_data = if let Some(skin) = skin {
        let accessor = skin.inverse_bind_matrices().ok_or_else(|| GltfError::SkinningError)?;
        let inv_bind_matrices = read_mat4_data(&accessor, blob_data)?;
        Some((skin, inv_bind_matrices))
    } else {
        None
    };

    for primitive in mesh.primitives() {
        if primitive.mode() != Mode::Triangles { return Err(GltfError::PrimitiveNotTriangles); }

        let mut positions = read_f32_vec3_data(&primitive.get(&Semantic::Positions).ok_or(GltfError::MissingPositions)?, blob_data)?;
        let mut normals_for_primitive = if let Some(accessor) = primitive.get(&Semantic::Normals) {
            has_any_normals = true;
            Some(read_f32_vec3_data(&accessor, blob_data)?)
        } else { None };
        
        let num_prim_vertices = positions.len();
        if num_prim_vertices == 0 { continue; }

        if let Some((skin, inv_bind_matrices)) = &skin_data {
            let joints_accessor = primitive.get(&Semantic::Joints(0)).ok_or_else(|| GltfError::SkinningError)?;
            let joints_u16 = match joints_accessor.data_type() {
                DataType::U8 => read_u8_vec4_data(&joints_accessor, blob_data)?.into_iter().map(|j| [j[0] as u16, j[1] as u16, j[2] as u16, j[3] as u16]).collect(),
                DataType::U16 => read_u16_vec4_data(&joints_accessor, blob_data)?,
                _ => return Err(GltfError::UnsupportedAccessorFormat),
            };
            let weights = read_f32_vec4_data(&primitive.get(&Semantic::Weights(0)).ok_or_else(|| GltfError::SkinningError)?, blob_data)?;

            for i in 0..num_prim_vertices {
                let joint_indices = joints_u16[i];
                let joint_weights = weights[i];
                
                let mut skin_matrix = Matrix4::zero();
                for j in 0..4 {
                    let joint_gltf_node_index = skin.joints().nth(joint_indices[j] as usize).ok_or_else(|| GltfError::SkinningError)?.index();
                    let weight = joint_weights[j];
                    if weight > 1e-6 {
                        let joint_transform = joint_global_transforms[joint_gltf_node_index];
                        let inv_bind_matrix = inv_bind_matrices[joint_indices[j] as usize];
                        skin_matrix = skin_matrix + (joint_transform * inv_bind_matrix) * weight;
                    }
                }
                
                let pos_pt = Point3::from(positions[i]);
                positions[i] = skin_matrix.transform_point(pos_pt).into();

                if let Some(ref mut normals) = normals_for_primitive {
                    if i < normals.len() {
                        let normal_vec = Vector3::from(normals[i]);
                        let normal_transform = skin_matrix.invert().map_or(Matrix4::identity(), |inv| inv.transpose());
                        let transformed_normal = (normal_transform * normal_vec.extend(0.0)).truncate().normalize();
                        normals[i] = transformed_normal.into();
                    }
                }
            }
        } else {
            for v_arr in positions.iter_mut() {
                *v_arr = node_transform_cg.transform_point(Point3::from(*v_arr)).into();
            }
            if let Some(ref mut prim_normals) = normals_for_primitive {
                let normal_transform = node_transform_cg.invert().map_or(Matrix4::identity(), |inv| inv.transpose());
                for n_arr in prim_normals.iter_mut() {
                    let transformed_normal = (normal_transform * Vector3::from(*n_arr).extend(0.0)).truncate().normalize();
                    *n_arr = transformed_normal.into();
                }
            }
        }
        aggregated_vertices.extend_from_slice(&positions);
        if let Some(prim_normals) = normals_for_primitive { aggregated_normals_vec.extend_from_slice(&prim_normals); }
        
        if let Some(indices_accessor) = primitive.indices() {
            aggregated_indices_vec.extend(read_indices_data(&indices_accessor, blob_data)?.into_iter().map(|idx| idx + current_vertex_offset));
        } else if num_prim_vertices % 3 == 0 { 
            aggregated_indices_vec.extend((0..num_prim_vertices as u32).map(|i| current_vertex_offset + i));
        }
        current_vertex_offset += num_prim_vertices as u32;
    } 
    
    if !aggregated_vertices.is_empty() && !has_any_normals { return Err(GltfError::MissingVertexNormals); }
    if has_any_normals && aggregated_normals_vec.len() < aggregated_vertices.len() { aggregated_normals_vec.resize(aggregated_vertices.len(), [0.0, 0.0, 1.0]); }
    let final_normals = if has_any_normals { Some(aggregated_normals_vec) } else { None };
    let has_vertices = !aggregated_vertices.is_empty();
    Ok(GltfMesh {
        name: mesh.name().map(String::from),
        vertices: aggregated_vertices,
        normals: final_normals,
        indices: if aggregated_indices_vec.is_empty() && has_vertices { None } else { Some(aggregated_indices_vec) },
    })
}

pub fn load_glb(glb_data: &[u8]) -> Result<Vec<GltfMesh>, GltfError> {
    let gltf = Gltf::from_slice(glb_data).map_err(GltfError::GltfImportError)?;
    let blob_data = gltf.blob.as_deref(); 
    let mut loaded_gltf_meshes = Vec::new();
    let nodes: Vec<_> = gltf.nodes().collect();
    
    let mut all_global_transforms = vec![Matrix4::identity(); nodes.len()];
    let mut stack: Vec<(usize, Matrix4<f32>)> = Vec::new();

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            stack.push((node.index(), Matrix4::identity()));
        }
    }
    
    while let Some((node_index, parent_transform)) = stack.pop() {
        let node = &nodes[node_index];
        let local_transform = Matrix4::from(node.transform().matrix());
        let global_transform = parent_transform * local_transform;
        all_global_transforms[node_index] = global_transform;

        for child in node.children() {
            stack.push((child.index(), global_transform));
        }
    }
    
    for node in &nodes {
        if let Some(mesh) = node.mesh() {
            let global_transform = all_global_transforms[node.index()];
            match process_mesh_data(mesh, global_transform, node.skin().as_ref(), blob_data, &all_global_transforms) {
                Ok(gltf_mesh_data) => {
                    if !gltf_mesh_data.vertices.is_empty() { 
                       loaded_gltf_meshes.push(gltf_mesh_data);
                    }
                }
                Err(e) => return Err(e), 
            }
        }
    }

    if loaded_gltf_meshes.is_empty() {
        let identity_matrix_cg = Matrix4::identity();
        let mut scened_mesh_indices = Vec::new();
        for node in &nodes { if let Some(mesh) = node.mesh() { scened_mesh_indices.push(mesh.index()); } }
        for (i, mesh) in gltf.meshes().enumerate() {
            if !scened_mesh_indices.contains(&i) {
                match process_mesh_data(mesh, identity_matrix_cg, None, blob_data, &all_global_transforms) {
                    Ok(gltf_mesh_data) => { if !gltf_mesh_data.vertices.is_empty() { loaded_gltf_meshes.push(gltf_mesh_data); } }
                    Err(e) => return Err(e), 
                 }
            }
        }
    }

    if loaded_gltf_meshes.is_empty() {
        return Err(GltfError::GltfImportError(gltf::Error::Validation(Vec::new()))); 
    }

    Ok(loaded_gltf_meshes)
}

pub fn convert_to_obj_data(gltf_meshes: &[GltfMesh]) -> obj::Obj<obj::Vertex, u32> {
    let mut final_obj_vertices: Vec<obj::Vertex> = Vec::new();
    let mut final_obj_indices: Vec<u32> = Vec::new();
    let mut current_vertex_offset: u32 = 0;

    for mesh in gltf_meshes {
        let default_normal = [0.0f32, 0.0f32, 1.0f32]; 

        for i in 0..mesh.vertices.len() {
            let pos = mesh.vertices[i];
            let normal_val = match &mesh.normals {
                Some(normals_vec) => {
                    if i < normals_vec.len() { normals_vec[i] } 
                    else { default_normal }
                }
                None => default_normal, 
            };
            final_obj_vertices.push(obj::Vertex {
                position: [pos[0], pos[1], pos[2]],
                normal: normal_val, 
            });
        }

        if let Some(indices_data) = &mesh.indices {
            for index in indices_data {
                final_obj_indices.push(*index + current_vertex_offset);
            }
        }
        current_vertex_offset += mesh.vertices.len() as u32;
    }

    obj::Obj {
        name: None, 
        vertices: final_obj_vertices,
        indices: final_obj_indices,
    }
}
