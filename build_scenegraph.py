# 一个非常棒的办法用来构建场景的拓扑图！

def build_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix

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
