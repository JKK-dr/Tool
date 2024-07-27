# clip相关的API

def get_img_feats(img, preprocess, clip_model):
    """
    Get the image features from the CLIP model
    :param img (np.array): the image to get the features from
    :param preprocess (torchvision.transforms): the preprocessing function
    :param clip_model (CLIP): the CLIP model
    :return: the image features
    """
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_in.cuda()).float()
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    img_feats = np.float32(img_feats.cpu())
    return img_feats

def get_imgs_feats(raw_imgs, preprocess, clip_model, clip_feat_dim):
    """
    Get the image features from the CLIP model for a list of images
    :param raw_imgs (list): the images to get the features from
    :param preprocess (torchvision.transforms): the preprocessing function
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :return: the image features
    """
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    for img_id, img in enumerate(raw_imgs):
        imgs_feats[img_id, :] = get_img_feats(img, preprocess, clip_model)
    return imgs_feats

def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    # in_text = ["a {} in the scene.".format(in_text)]
    text_tokens = open_clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats

def match_text_to_imgs(language_instr, images_list):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    imgs_feats = get_imgs_feats(images_list)
    text_feats = get_text_feats([language_instr])
    scores = imgs_feats @ text_feats.T
    scores = scores.squeeze()
    return scores, imgs_feats, text_feats
