#include "pcFuse.h"

//构造函数
CpcFuse::CpcFuse(void)
{
	// Create a PCLVisualizer object
	p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
	pv = new pcl::visualization::PCLVisualizer("Show Result");
	p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
	p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);
	pv->createViewPort(0.0, 0.0, 1.0, 1.0, vp_3);
}
//析构函数
CpcFuse::~CpcFuse(void)
{
	delete p;
	delete pv;
}

void CpcFuse::voxelFilter(PointCloud::Ptr &cloud_in, PointCloud::Ptr &cloud_out, float gridsize){
	pcl::VoxelGrid<PointT> vox_grid;
	vox_grid.setLeafSize(gridsize, gridsize, gridsize);
	vox_grid.setInputCloud(cloud_in);
  vox_grid.filter(*cloud_out);
  cout << "PointCloud before voxelfiltering: " << cloud_in->width * cloud_in->height 
  << " data points (" << pcl::getFieldsList (*cloud_in) << ")."<<endl;
  cout << "PointCloud after voxelfiltering: " << cloud_out->width * cloud_out->height 
  << " data points (" << pcl::getFieldsList (*cloud_out) << ").\n"<<endl;
	return;
}

pcl::PointCloud<pcl::Normal>::Ptr CpcFuse::getNormals(PointCloud::Ptr cloud, double radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normalsPtr (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointT,pcl::Normal> norm_est;
    norm_est.setInputCloud(cloud);
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(radius);
    //norm_est.setRadiusSearch(radius);
    norm_est.compute(*normalsPtr);
    return normalsPtr;

}

FPFHCloud::Ptr CpcFuse::getFeatures(PointCloud::Ptr cloud,pcl::PointCloud<pcl::Normal>::Ptr normals,double radius)
{
    FPFHCloud::Ptr features (new FPFHCloud);    
    pcl::FPFHEstimationOMP<PointT,pcl::Normal,FPFHT> fpfh_est;
    fpfh_est.setNumberOfThreads(4);
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchMethod(tree);
    fpfh_est.setKSearch(radius);
   // fpfh_est.setRadiusSearch(radius);
    fpfh_est.compute(*features);
    return features;
}
//提取SIFT关键点
pcl::PointCloud<PointT>::Ptr CpcFuse::getSift(PointCloud::Ptr cloud)
{
	pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;
	sift.setInputCloud(cloud);//设置输入点云 
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());  
	sift.setSearchMethod(tree);//创建一个空的kd树对象tree，并把它传递给sift检测对象  
	sift.setScales(0.01, 6, 4);//指定搜索关键点的尺度范围  
	sift.setMinimumContrast(0.01);//设置限制关键点检测的阈值 
	pcl::PointCloud<pcl::PointWithScale> result;
	sift.compute(result);//执行sift关键点检测，保存结果在result   
	pcl::PointCloud<PointT>::Ptr cloud_temp (new pcl::PointCloud<PointT>); 
	pcl::copyPointCloud(result, *cloud_temp);//
	return cloud_temp;
}


//sac_ia配准
void CpcFuse::sac_ia_align(PointCloud::Ptr source,PointCloud::Ptr target,PointCloud::Ptr finalcloud,Eigen::Matrix4f *init_transform,
   int max_sacia_iterations,double min_correspondence_dist,double max_correspondence_dist)
{

  vector<int> indices1;
  vector<int> indices2;
  PointCloud::Ptr sourceds (new PointCloud);
  PointCloud::Ptr targetds (new PointCloud);
  pcl::removeNaNFromPointCloud(*source,*source,indices1);
  pcl::removeNaNFromPointCloud(*target,*target,indices2);
//降采样
  voxelFilter(source,sourceds,VOXEL_GRID_SIZE);  
  voxelFilter(target,targetds,VOXEL_GRID_SIZE);  

 //提取Sift关键点
 // sourceds = getSift(source);
 // targetds = getSift(target);
  cout<<"1:extracting keypoints"<<endl;  
//计算法向量
  pcl::PointCloud<pcl::Normal>::Ptr source_normal (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>::Ptr target_normal (new pcl::PointCloud<pcl::Normal>);
  source_normal=getNormals(sourceds,radius_normal);
  target_normal=getNormals(targetds,radius_normal);
  cout<<"2:computing normals"<<endl;
//计算FPFH特征    
  FPFHCloud::Ptr source_feature (new FPFHCloud);
  FPFHCloud::Ptr target_feature (new FPFHCloud);
  source_feature=getFeatures(sourceds,source_normal,radius_feature);
  target_feature=getFeatures(targetds,target_normal,radius_feature);
  cout<<"3:computing feature descriptors"<<endl;

//SAC-IA配准  
  pcl::SampleConsensusInitialAlignment<PointT,PointT,FPFHT> sac_ia;
  Eigen::Matrix4f final_transformation;
  sac_ia.setInputSource(targetds);
  sac_ia.setSourceFeatures(target_feature);
  sac_ia.setInputTarget(sourceds);
  sac_ia.setTargetFeatures(source_feature);
  sac_ia.setMaximumIterations(max_sacia_iterations); //RANSAC迭代次数
  sac_ia.setMinSampleDistance(min_correspondence_dist);
  sac_ia.setMaxCorrespondenceDistance(max_correspondence_dist);
  
  PointCloud::Ptr output (new PointCloud);
  timecal.tic();  
  sac_ia.align(*output);
  cout<<"Finished SAC_IA Initial Regisration in "<<timecal.toc()<<"ms\n"<<endl;
  *init_transform=sac_ia.getFinalTransformation();
  pcl::transformPointCloud(*target,*finalcloud,*init_transform);
}
////////////////////////////////
//显示配准结果，刚性变换矩阵
void CpcFuse::print4x4Matrix(Eigen::Matrix4f &matrix)
{
  
  printf ("Rotation matrix:\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the first viewport of the visualizer
 *
 */
void CpcFuse::showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);

 // PCL_INFO ("Press q to begin the registration.\n");
 // p-> spin();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
 *
 */
void CpcFuse::showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source");
  p->removePointCloud ("target");


  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");


  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

  p->spinOnce();
}

void CpcFuse::showCloudsFinal(const PointCloud::Ptr cloud, bool once=true)
{
	pv->removePointCloud("final");
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> color_handler(cloud); // 按照RGB字段进行渲染
	pv->addPointCloud(cloud, color_handler, "final", vp_3); //添加点云
	pv->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "final"); // 设置点云大小
	if (once) {
		pv->spinOnce();
	}
	else {
		pv->spin();
	}
	
}

////////////////////////////////////////////////////////////////////////////////
/** \brief Load a set of PCD files that we want to register together
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param models the resultant vector of point cloud datasets
  */
  //扫描文件夹
void CpcFuse::GetFilesFromDirectory(std::vector<std::string> &files, const char *directoryPath)
{
	struct _finddata_t fileinfo;
	intptr_t hFile = 0;
	char tmpPath[MAX_PATH] = { 0 };
	sprintf_s(tmpPath, "%s\\*.txt", directoryPath);
	//cout << tmpPath << endl;
	if ((hFile = _findfirst(tmpPath, &fileinfo)) == -1) { return; }
	do
	{
		if ((fileinfo.attrib &  _A_SUBDIR))
		{
			if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
			{
				sprintf_s(tmpPath, "%s\\%s", directoryPath, fileinfo.name);
				GetFilesFromDirectory(files, tmpPath);
			}
		}
		else
		{
			sprintf_s(tmpPath, "%s\\%s", directoryPath, fileinfo.name);
			files.push_back(tmpPath);
		}
	} while (_findnext(hFile, &fileinfo) == 0);
	
	_findclose(hFile);
}

//txt格式点云文件转换为pcd格式
void CpcFuse::txt2pcd(const char *filename, const PointCloud::Ptr cloud) {

	typedef struct tagPOINT_3D
	{
		double x;  //mm world coordinate x  
		double y;  //mm world coordinate y  
		double z;  //mm world coordinate z  
		double r; //frgb
		double g;
		double b;
	}POINT_WORLD;
	int number_Txt;
	FILE *fp_txt;
	tagPOINT_3D TxtPoint;
	vector<tagPOINT_3D> m_vTxtPoints;
	fp_txt = fopen(filename, "r");
	if (fp_txt)
	{
		while (fscanf(fp_txt, "%lf %lf %lf %lf %lf %lf", &TxtPoint.x, &TxtPoint.y, &TxtPoint.z, &TxtPoint.r, &TxtPoint.g, &TxtPoint.b) != EOF)
		{
			m_vTxtPoints.push_back(TxtPoint);
		}
	}
	else {
		std::cout << "加载失败" << endl;
	}

	number_Txt = m_vTxtPoints.size();

	//这里使用“PointXYZ”是因为我后面给的点云信息是包含的三维坐标，同时还有点云信息包含的rgb颜色信息的或者还有包含rgba颜色和强度信息。

	// Fill in the cloud data  

	cloud->width = number_Txt;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t t = 0; t < cloud->points.size(); ++t)
	{
		int frgb = 0;
		cloud->points[t].x = m_vTxtPoints[t].x;
		cloud->points[t].y = m_vTxtPoints[t].y;
		cloud->points[t].z = m_vTxtPoints[t].z;
		frgb = ((int)m_vTxtPoints[t].r << 16 | (int)m_vTxtPoints[t].g << 8 | (int)m_vTxtPoints[t].b);
		cloud->points[t].rgb = *reinterpret_cast<float*>(&frgb);//frgb在这由int型转换成了float型。
					  //reinterpret_cast()为指针类型转换。
	}
}

int CpcFuse::loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
	//argc<2 扫描文件夹 argc>=2 读取点云文件
	//确定文件格式，如果是pcd直接读取，如果是txt则转换格式
	if (argc == 2) {
		//获取当前工作目录
		char filepath[255];
		_getcwd(filepath, sizeof(filepath));
		//扫描指定文件夹
		strcat(filepath,argv[1]);
		//cout << filepath << endl;
		std::vector<std::string> filename;
		GetFilesFromDirectory(filename,filepath);
		//读取文件
		if (filename.empty())
		{
			PCL_ERROR("No files found in the directory!");
			return (-1);
		}
		for (size_t i = 0; i < filename.size(); ++i)
		{
			const char *txt_file = filename[i].data();
			PCD m;
			m.f_name = filename[i];
			txt2pcd(txt_file, m.cloud);
			//remove NAN points from the cloud
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(*m.cloud, *m.cloud, indices);

			models.push_back(m);
		}

	}
	else {
		std::string extension(".pcd");
		// Suppose the first argument is the actual test model
		for (int i = 1; i < argc; i++)
		{
			std::string fname = std::string(argv[i]);
			// Needs to be at least 5: .plot
			if (fname.size() <= extension.size())
				continue;

			std::transform(fname.begin(), fname.end(), fname.begin(), (int(*)(int))tolower);

			//check that the argument is a pcd file
			if (fname.compare(fname.size() - extension.size(), extension.size(), extension) == 0)
			{
				// Load the cloud and saves it into the global list of models
				PCD m;
				m.f_name = argv[i];
				pcl::io::loadPCDFile(argv[i], *m.cloud);
				//remove NAN points from the cloud
				std::vector<int> indices;
				pcl::removeNaNFromPointCloud(*m.cloud, *m.cloud, indices);

				models.push_back(m);
			}
		}
	}
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
void CpcFuse::pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.02, 0.02, 0.02);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);  //
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[7] = { 1.0, 1.0, 1.0, 0.005, 0.005, 0.005, 1.0 }; //rgb
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-6);//6
  reg.setEuclideanFitnessEpsilon(0.001);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (0.1);  
  //reg.RANSACOutlierRejectionThreshold(1.5);
  reg.setMaximumIterations (300);//30
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);

  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  //reg.hasConverged
  for (int i = 0; i < 200; ++i)
  {    
	//  PCL_INFO("Press q to continue the registration.\n");
	//  p->spin();
    points_with_normals_src = reg_result;// save cloud for visualization purpose
    reg.setInputSource (points_with_normals_src);
   // timecal.tic();
    reg.align (*reg_result);	
    //cout<<"Applied %ICP "<<time.toc()<<" ms\n";	
   // PCL_INFO("TIME:Applied num %d ICP in %f ms\n",i+1,timecal.toc());
	//cout << "has conveged:" << reg.hasConverged()  << endl;
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
	//if(reg.hasConverged())

  }

  // Get the transformation from target to source
  targetToSource = Ti.inverse();
 

  final_transform = targetToSource;
  std::cout << "\nICP has converged, score is " << reg.getFitnessScore () << std::endl;
  //std::cout <<icp.getFinalTransformation ()<<std::endl;
  //print4x4Matrix(final_transform);

  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  p->removePointCloud ("source");
  p->removePointCloud ("target");

  PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 255, 0, 0);
  PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 0, 255, 0);
  p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

	//PCL_INFO ("Press q to continue the registration.\n");
  //p->spin ();

  p->removePointCloud ("source"); 
  p->removePointCloud ("target");

  //add the source to the transformed target
  //*output += *cloud_src;
  
  
 }
 //重采样
 void CpcFuse::resampling(const PointCloud::Ptr cloud_src, pcl::PointCloud<PointNormalT> &output,bool setComputeNormals=false,bool setPolynomialFit=false)
 {
	 //移动最小二乘重采样
	 pcl::MovingLeastSquares<PointT, PointNormalT> mls;//移动最小二乘法计算法向

	 mls.setComputeNormals(setComputeNormals);// 设置在最小二乘计算中需要进行法线估计,不需要可跳过
	 // Set parameters
	 mls.setInputCloud(cloud_src);
	 mls.setPolynomialFit(setPolynomialFit);  //多项式拟合提高精度,可false 加快速度,或选择其他来控制平滑过程
	 mls.setSearchMethod(tree);

	 mls.setSearchRadius(0.01);
	 // Reconstruct
	 timecal.tic();
	 PCL_INFO("resampling...");
	 mls.process(output);
	 cout << "Finished in " << timecal.toc() << "ms\n" << endl;
 }
 
 //配对点云，更新点云
 void CpcFuse::pairwise_incremental_registration(const PointCloud::Ptr source, const PointCloud::Ptr target, PointCloud::Ptr final, Eigen::Matrix4f &final_transform)
 {
	 Eigen::Matrix4f pairTransform, localTransform;
	 // Add visualization data
	 showCloudsLeft(source, target);
	 PointCloud::Ptr temp(new PointCloud);
	 PointCloud::Ptr result(new PointCloud);
	 PointCloud::Ptr show(new PointCloud);

	 PointCloud::Ptr init_result(new PointCloud);
	 //  *init_result = *target;
	 Eigen::Matrix4f init_transform = Eigen::Matrix4f::Identity(); //用单位阵初始化
	 //Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
	 
	 sac_ia_align(source, target, init_result, &init_transform, max_sacia_iterations, min_correspondence_dist, max_correspondence_dist); //粗配准
	 pairAlign(source, init_result, temp, pairTransform, true); //细配准
	
	 print4x4Matrix(init_transform);
	 print4x4Matrix(pairTransform);
	 localTransform = pairTransform * init_transform;

	 pcl::transformPointCloud(*temp, *result, final_transform);
	 
	 //更新点云
	 *final += *result;
	 final_transform = final_transform * localTransform;
	 //GlobalTransform = GlobalTransform * init_transform;
	 //显示
	 *show = *final;
	 bool showonce = true;
	 pcl::transformPointCloud(*show, *show, final_transform.inverse());
	 showCloudsFinal(show, showonce);
 }
 //读取文件夹循环
 int CpcFuse::pcFuse(int argc, char** argv)
 {
	 // Load data
	 std::vector<CpcFuse::PCD, Eigen::aligned_allocator<CpcFuse::PCD> > data;
	 loadData(argc, argv, data);

	 // Check user input
	 if (data.empty())
	 {
		 PCL_ERROR("Syntax is: %s <source.pcd> <target.pcd> [*]", argv[0]);
		 return (-1);
	 }
	 PCL_INFO("Loaded %d datasets.", (int)data.size());



	 PointCloud::Ptr final(new PointCloud), source, target;
	 PointCloud::Ptr show(new PointCloud);
	 Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity();

	 final = data[0].cloud;

	 for (size_t i = 1; i < data.size(); ++i)
	 {
		 source = data[i - 1].cloud;
		 target = data[i].cloud;

		 pairwise_incremental_registration(source, target, final, GlobalTransform);
		 //save aligned pair, transformed into the first cloud's frame
		 std::stringstream ss;
		 ss << i << ".pcd";
		 pcl::io::savePCDFile(ss.str(), *final, true);

		 //移动最小二乘重采样
		 if (i == data.size() - 1) {
			 //先降采样再重采样
			 PointCloud::Ptr temp_cloud(new PointCloud);
			 voxelFilter(final, temp_cloud, 0.002);
			 pcl::PointCloud<PointNormalT> mls_points;
			 resampling(temp_cloud, mls_points, false, false);
			 pcl::io::savePCDFile("result.pcd", mls_points, true);
			 copyPointCloud(mls_points, *show);
			 pcl::transformPointCloud(*show, *show, GlobalTransform.inverse());
			 showCloudsFinal(show, false);
		 }


	 }
 }



