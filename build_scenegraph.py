# 一个非常棒的办法用来构建场景的拓扑图！

class DetectionList(list):
    def get_values(self, key, idx:int=None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)
    
    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self
    
    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self
    
    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes
    
    def color_by_most_common_classes(self, colors_dict: dict[str, list[float]], color_bbox: bool=True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color
                
    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return
        
        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]
            
    
class MapObjectList(DetectionList):
    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor 
        new_clip_ft = to_tensor(new_clip_ft)
        
        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities
    
    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            
            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])
            
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            
            s_obj_list.append(s_obj_dict)
            
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)
            
            new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
            new_obj['text_ft'] = to_tensor(new_obj['text_ft'])
            
            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
            
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']
            
            self.append(new_obj)

def compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < cfg.downsample_voxel_size ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def build_scenegraph(args):
    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    response_dir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses = []
    object_tags = []
    also_indices_to_remove = [] # indices to remove if the json file does not exist
    for idx in range(len(scene_map)):
        # check if the json file exists first 
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])

    # Remove segments that correspond to "invalid" tags
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    # Also remove segments that do not have a minimum number of observations
    indices_to_remove = set(indices_to_remove)
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # Remove objects with less than args.min_views_per_object observations
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    indices_to_remove = list(indices_to_remove)
    # combine with also_indices_to_remove and sort the list
    indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
    
    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments")
    
    # Filtering responses based on segment_ids_to_retain
    responses = [resp for resp in responses if resp['id'] in segment_ids_to_retain]

    # Assuming each response dictionary contains an 'object_tag' key for the object tag.
    # Extract filtered object tags based on filtered_responses
    object_tags = [resp['response']['object_tag'] for resp in responses]


    pruned_scene_map = []
    pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        pruned_object_tags.append(object_tags[_idx])
    scene_map = MapObjectList(pruned_scene_map)
    object_tags = pruned_object_tags
    del pruned_scene_map
    # del pruned_object_tags
    gc.collect()
    num_segments = len(scene_map)

    for i in range(num_segments):
        scene_map[i]["caption_dict"] = responses[i]
        # scene_map[i]["object_tag"] = object_tags[i]

    # Save the pruned scene map (create the directory if needed)
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix(args, scene_map)

    # Construct a weighted adjacency matrix based on similarity scores
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        TIMEOUT = 25  # timeout in seconds

        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx1],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx2],
                        },
                    }
                    print(f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}")

                    relation_queries.append(input_dict)

                    input_json_str = json.dumps(input_dict)

                    # Default prompt
                    DEFAULT_PROMPT = """
                    The input is a list of JSONs describing two objects "object1" and "object2". You need to produce a JSON
                    string (and nothing else), with two keys: "object_relation", and "reason".

                    Each of the JSON fields "object1" and "object2" will have the following fields:
                    1. bbox_extent: the 3D bounding box extents of the object
                    2. bbox_center: the 3D bounding box center of the object
                    3. object_tag: an extremely brief description of the object

                    Produce an "object_relation" field that best describes the relationship between the two objects. The
                    "object_relation" field must be one of the following (verbatim):
                    1. "a on b": if object a is an object commonly placed on top of object b
                    2. "b on a": if object b is an object commonly placed on top of object a
                    3. "a in b": if object a is an object commonly placed inside object b
                    4. "b in a": if object b is an object commonly placed inside object a
                    5. "none of these": if none of the above best describe the relationship between the two objects

                    Before producing the "object_relation" field, produce a "reason" field that explains why
                    the chosen "object_relation" field is the best.
                    """

                    start_time = time.time()
                    chat_completion = openai.ChatCompletion.create(
                        # model="gpt-3.5-turbo",
                        model="gpt-4",
                        messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                        timeout=TIMEOUT,  # Timeout in seconds
                    )
                    elapsed_time = time.time() - start_time
                    output_dict = input_dict
                    if elapsed_time > TIMEOUT:
                        print("Timed out exceeded!")
                        output_dict["object_relation"] = "FAIL"
                        continue
                    else:
                        try:
                            # Attempt to parse the output as a JSON
                            chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
                            # If the output is a valid JSON, then add it to the output dictionary
                            output_dict["object_relation"] = chat_output_json["object_relation"]
                            output_dict["reason"] = chat_output_json["reason"]
                        except:
                            output_dict["object_relation"] = "FAIL"
                            output_dict["reason"] = "FAIL"
                    relations.append(output_dict)

                    # print(chat_completion["choices"][0]["message"]["content"])

            # Save the query JSON to file
            print("Saving query JSON to file...")
            with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    scenegraph_edges = []

    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
            if relations[_idx]["object_relation"] != "none of these":
                scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
            _idx += 1
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)
