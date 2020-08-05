#pragma once
#define BOOST_TYPEOF_EMULATION
#include <boost/make_shared.hpp> //共享指针
#include <iostream>
#include <string>
#include<direct.h>
#include<io.h>
#include <tchar.h>

#include <boost/make_shared.hpp>
#include <pcl/console/time.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
//filter
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>//包含fpfh加速计算的omp多核并行计算
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/common/transforms.h>
#include <pcl/surface/mls.h>   //最小二乘平滑处理类定义

#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointXYZRGBNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

typedef pcl::FPFHSignature33 FPFHT;
typedef pcl::PointCloud<FPFHT> FPFHCloud;

class CpcFuse {
public:
	pcl::visualization::PCLVisualizer *p;
	pcl::visualization::PCLVisualizer *pv;
	//its left and right viewports
	int vp_1, vp_2, vp_3;
	pcl::console::TicToc timecal;
	pcl::search::KdTree<PointT>::Ptr tree;
	//convenient structure to handle our pointclouds
	struct PCD
	{
		PointCloud::Ptr cloud;
		std::string f_name;

		PCD() : cloud(new PointCloud) {};
	};

	struct PCDComparator
	{
		bool operator () (const PCD& p1, const PCD& p2)
		{
			return (p1.f_name < p2.f_name);
		}
	};


	// Define a new point representation for < x, y, z, curvature >
	class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
	{
		using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

	public:
		MyPointRepresentation()
		{
			// Define the number of dimensions
			nr_dimensions_ = 7;
		}

		// Override the copyToFloatArray method to define our feature vector
		virtual void copyToFloatArray(const PointNormalT &p, float * out) const
		{
			// < x, y, z, curvature >
			out[0] = p.x;
			out[1] = p.y;
			out[2] = p.z;
			float frgb = p.rgb;
			int nrgb = *reinterpret_cast<int*>(&frgb);//float型转换int型
			out[3] = (nrgb >> 16) & 0x0000ff;
			out[4] = (nrgb >> 8) & 0x0000ff;
			out[5] = (nrgb) & 0x0000ff;
			out[6] = p.curvature;
		}
	};

	CpcFuse(void);
	~CpcFuse(void);
	void voxelFilter(PointCloud::Ptr &cloud_in, PointCloud::Ptr &cloud_out, float gridsize);
	pcl::PointCloud<pcl::Normal>::Ptr getNormals(PointCloud::Ptr cloud, double radius);
	FPFHCloud::Ptr getFeatures(PointCloud::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, double radius);
	pcl::PointCloud<PointT>::Ptr getSift(PointCloud::Ptr cloud);
	void sac_ia_align(PointCloud::Ptr source, PointCloud::Ptr target, PointCloud::Ptr finalcloud, Eigen::Matrix4f *init_transform,
		int max_sacia_iterations, double min_correspondence_dist, double max_correspondence_dist);
	void print4x4Matrix(Eigen::Matrix4f &matrix);
	void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source);
	void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source);
	void showCloudsFinal(const PointCloud::Ptr cloud, bool once);
	void GetFilesFromDirectory(std::vector<std::string> &files, const char *directoryPath);
	void txt2pcd(const char *filename, const PointCloud::Ptr cloud);
	int loadData(int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models);
	void pairAlign(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample);
	void resampling(const PointCloud::Ptr cloud_src, pcl::PointCloud<PointNormalT> &output, bool setComputeNormals, bool setPolynomialFit);
	void pairwise_incremental_registration(const PointCloud::Ptr source, const PointCloud::Ptr target, PointCloud::Ptr final, Eigen::Matrix4f &final_transform);
	int pcFuse(int argc, char** argv);


private:
	const float VOXEL_GRID_SIZE = 0.05;//
	const double radius_normal = 100;//
	const double radius_feature = 100;
	const double max_sacia_iterations = 1000;
	const double min_correspondence_dist = 0.01;
	const double max_correspondence_dist = 1000;
};