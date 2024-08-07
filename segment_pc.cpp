// 代码来自https://github.com/PJLab-ADG/SensorsCalibration的lidar2camera部分（auto_calib）,可用于点云数据预处理【传统方案】
void Calibrator::Segment_pc(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                            pcl::PointCloud<pcl::Normal>::Ptr normals,
                            std::vector<pcl::PointIndices> &seg_indices)
{   
    // compute_normals
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> norm_est;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZI>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(40);
    // norm_est.setRadiusSearch(5);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);

    // plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr indices_plane(new pcl::PointIndices);
    pcl::SACSegmentationFromNormals<pcl::PointXYZI, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.2);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(3000);
    seg.setDistanceThreshold(0.2);
    seg.setInputCloud(cloud);
    seg.setInputNormals(normals);
    seg.segment(*indices_plane, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZI> extract(true);
    extract.setInputCloud(cloud);
    pcl::PointIndices::Ptr indices_notplane(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>);
    int plane_size = indices_plane->indices.size();
    pcl::PointIndices::Ptr indices_plane_all(new pcl::PointIndices);
    while (plane_size > 2000)
    {
        std::cout << "Plane points: " << plane_size << std::endl;
        seg_indices.push_back(*indices_plane);
        seg_point_num_.push_back(plane_size);
        indices_plane_all->indices.insert(indices_plane_all->indices.end(), indices_plane->indices.begin(), indices_plane->indices.end());
        extract.setIndices(indices_plane_all);
        extract.filter(*cloud_out);
        extract.getRemovedIndices(*indices_notplane);
        seg.setIndices(indices_notplane);
        seg.segment(*indices_plane, *coefficients);
        plane_size = indices_plane->indices.size();
    }
    std::cout << "Plane points < 1500, stop extracting plane." << std::endl;

    // euclidean cluster extraction
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    std::vector<pcl::PointIndices> eu_cluster_indices;
    ec.setClusterTolerance(0.25);
    ec.setMaxClusterSize(10000);
    ec.setMinClusterSize(50);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.setIndices(indices_notplane);
    ec.extract(eu_cluster_indices);
    std::cout << "Euclidean cluster number: " << eu_cluster_indices.size() << std::endl;

    seg_indices.insert(seg_indices.end(), eu_cluster_indices.begin(), eu_cluster_indices.end());
    for (auto it = eu_cluster_indices.begin(); it != eu_cluster_indices.end(); it++)
    {
        seg_point_num_.push_back((*it).indices.size());
    }

    N_SEG = seg_indices.size();
    std::cout << "Extract " << N_SEG << " segments from point cloud." << std::endl;
}
