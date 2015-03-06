/** ****************************************************************************
 *  @file    FaceForest.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/06
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceForest.hpp>
#include <boost/filesystem.hpp>
#include <face_utils.hpp>
#include <MeanShift.hpp>

FaceForest::FaceForest
  (
  FaceForestOptions option
  ) :
    m_ff_options(option)
{
  //loading face cascade
  CV_Assert(load_face_cascade(option.face_detection_option.path_face_cascade));

  //loading head pose forest
  m_hp_forest.load(option.hp_forest_param.treePath,
      option.hp_forest_param);

  //loading ffd trees
  num_trees = option.mp_forest_param.nTrees;

  m_mp_forest.setParam(option.mp_forest_param);
  get_paths_to_trees(option.mp_forest_param.treePath, m_ff_options.mp_tree_paths);

  loading_all_trees(m_ff_options.mp_tree_paths);
  is_inizialized = true;
};

void
FaceForest::detect_face
  (
  const cv::Mat &img,
  cv::CascadeClassifier &face_cascade,
  FaceDetectionOption option,
  std::vector<cv::Rect> &faces
  )
{
  int flags = 0; //CV_HAAR_SCALE_IMAGE;
  cv::Size minFeatureSize = cv::Size(option.min_feature_size, option.min_feature_size);

  // Detect faces
  face_cascade.detectMultiScale(img, faces, option.search_scale_factor, option.min_neighbors, flags, minFeatureSize);

  // The face detection bboxes are too tight for us
  for (unsigned int i = 0; i < faces.size(); i++)
  {
    int off_set_x = faces[i].width * 0.05;
    int off_set_y = faces[i].width * 0.15;

    cv::Rect roi = intersect(cv::Rect(faces[i].x - off_set_x, faces[i].y,
        faces[i].width + off_set_x * 2, faces[i].height + off_set_y * 2),
        cv::Rect(0, 0, img.cols, img.rows));
    faces[i] = roi;
  }
};

void
FaceForest::estimate_head_pose
  (
  const ImageSample &img_sample,
  const cv::Rect &face_bbox,
  const Forest<HeadPoseSample> &forest,
  HeadPoseEstimatorOption option,
  float *head_pose,
  float *head_pose_variance
  )
{
  // Collect leafs
  std::vector<HeadPoseLeaf*> leafs;
  get_headpose_votes_mt(img_sample, forest, face_bbox, leafs, option.step_size);

  // Parse leaf
  int n = 0;
  float sum = 0;
  float sum_sq = 0;
  for (unsigned int j = 0; j < leafs.size(); ++j)
  {
    if (leafs[j]->forgound > option.min_forground_probability)
    {
      float m = 0;
      for (int ii = 0; ii < option.num_head_pose_labels; ii++)
      {
        m += leafs[j]->hist_labels[ii] * ii;
      }
      m /= (leafs[j]->nSamples * leafs[j]->forgound);
      sum += m;
      sum_sq += m * m;
      n++;
    }
  }
  double mean = static_cast<float> (sum) / n;
  double variance = static_cast<float> (sum_sq) / n - mean * mean;

  const float norm_factor = 0.05;
  variance *= norm_factor;
  mean -= 2;

  *head_pose = mean;
  *head_pose_variance = variance;
};

void
FaceForest::estimate_ffd
  (
  const ImageSample &image_sample,
  const cv::Rect face_bbox,
  const Forest<MPSample> &mp_forest,
  MultiPartEstimatorOption option,
  std::vector<cv::Point> &ffd_cordinates
  )
{
  int num_parts = option.num_parts;
  ffd_cordinates.clear();
  ffd_cordinates.resize(num_parts);
  std::vector < std::vector<Vote> > votes(num_parts);

  get_ffd_votes_mt(image_sample, mp_forest, face_bbox, votes, option);

  //	boost::thread_pool::executor pool;
  //	for (int j= 0; j < num_parts; j++){
  //		pool.submit(boost::bind(&MeanShift::shift, votes[j], ffd_cordinates[j],10,3));
  //	}
  //	pool.join_all();

  MeanShiftOption ms_option;
  for (int j = 0; j < num_parts; j++)
    MeanShift::shift(votes[j], ffd_cordinates[j], ms_option);

  //	std::vector<cv::Point> gt;
  //	plot_ffd_votes(image_sample.featureChannels[0], votes,
  //			ffd_cordinates, gt);
};

void
FaceForest::show_results
  (
  const cv::Mat img,
  std::vector<Face> &faces,
  upm::Viewer &viewer
  )
{
  cv::Mat image = img.clone();

  for (unsigned int j = 0; j < faces.size(); j++) {
    for (unsigned int ii = 0; ii < faces[j].ffd_cordinates.size(); ii++)
    {
      int x = faces[j].ffd_cordinates[ii].x + faces[j].bbox.x;
      int y = faces[j].ffd_cordinates[ii].y + faces[j].bbox.y;
      viewer.circle(x, y, 3, -1, cv::Scalar(0,255,0));
    }
    cv::Rect bbox = faces[j].bbox;
    cv::Point_<int> a(bbox.x, bbox.y);
    cv::Point_<int> b(bbox.x + bbox.width, bbox.y);
    cv::Point_<int> c(bbox.x, bbox.y + bbox.height);
    cv::Point_<int> d(bbox.x + bbox.width, bbox.y + bbox.height);

    cv::Point_<int> y1(bbox.x, bbox.y + (bbox.height / 2));
    cv::Point_<int> y2(bbox.x + bbox.width, bbox.y + bbox.height / 2);
    float yaw = faces[j].headpose * 80;
    if (yaw < 0) {
      y2.x -= yaw;
    } else {
      y1.x -= yaw;
    }
    cv::Scalar color(0, 0, 255);
    viewer.line(a.x, a.y, y1.x, y1.y, 2, color);
    viewer.line(y1.x, y1.y, c.x, c.y, 2, color);
    viewer.line(b.x, b.y, y2.x, y2.y, 2, color);
    viewer.line(y2.x, y2.y, d.x, d.y, 2, color);
    viewer.line(a.x, a.y, b.x, b.y, 2, color);
    viewer.line(c.x, c.y, d.x, d.y, 2, color);
  }
};

void FaceForest::analize_image
  (
  cv::Mat img,
  std::vector<Face> &faces
  )
{
  CV_Assert(is_inizialized);

  //detect the face
  std::vector < cv::Rect > faces_bboxes;
  detect_face(img, m_face_cascade, m_ff_options.face_detection_option, faces_bboxes);

  std::cout << faces_bboxes.size() << " detected faces." << std::endl;
  // for each detected face
  for (unsigned int i = 0; i < faces_bboxes.size(); i++) {
    Face f;
    analize_face(img, faces_bboxes[i], f);
    faces.push_back(f);
  }
};

void
FaceForest::analize_face
  (
  const cv::Mat img,
  cv::Rect face_bbox,
  Face &result_face,
  bool normalize
  )
{
  CV_Assert(is_inizialized);
  CV_Assert(img.type() == CV_8UC1);
  result_face.bbox = face_bbox;

  ForestParam param = m_ff_options.hp_forest_param;
  // Rescale and extract face
  cv::Mat face;
  float scale = static_cast<float> (param.faceSize) / face_bbox.width;
  cv::resize(img(face_bbox), face, cv::Size(face_bbox.width * scale,
      face_bbox.height * scale), 0, 0);

  if (normalize) {
    equalizeHist(face, face);
  }

  Timing timer;
  timer.start();
  ImageSample sample(face, param.features, fcf, true);
  //	cout << "creating img sample: " << timer.elapsed() << endl;
  timer.restart();
  //detect head pose
  int hist_size = 5;
  float headpose = 0;
  float variance = 0;
  estimate_head_pose(sample, cv::Rect(0, 0, face.cols, face.rows), m_hp_forest,
                     m_ff_options.pose_estimator_option, &headpose, &variance);

  result_face.headpose = headpose;
  // compute area under curve
  std::vector<float> poseT(hist_size + 1);
  poseT[0] = -2.5;
  poseT[1] = -0.35;
  poseT[2] = -0.20;
  poseT[3] = -poseT[2];
  poseT[4] = -poseT[1];
  poseT[5] = -poseT[0];

  std::vector<float> pose_freq(hist_size);
  int max_area = 0;
  int dominant_headpose = 0;
  for (int j = 0; j < hist_size; j++) {
    float area = areaUnderCurve(poseT[j], poseT[j + 1], headpose,
        sqrt(variance));
    pose_freq[j] = area;
    if (max_area < area) {
      max_area = area;
      dominant_headpose = j;
    }
  }

  timer.restart();

  //add new tree based on the estimated heas_pose
  CV_Assert(m_trees.size() == pose_freq.size());
  CV_Assert(static_cast<int> (m_trees.size()) == hist_size);

  m_mp_forest.cleanForest();

  for (unsigned int i = 0; i < m_trees.size(); i++) {
    int n_trees = pose_freq[i] * num_trees;
    for (int j = 0; j < n_trees; j++) {
      m_mp_forest.addTree(m_trees[i][j]);
    }
  }
  // correcting rounding errors
  for (int i = m_mp_forest.numberOfTrees(); i < num_trees; i++) {
    m_mp_forest.addTree(m_trees[dominant_headpose][i]);
  }

  //estimate ffd
  estimate_ffd(sample, cv::Rect(0, 0, face.cols, face.rows), m_mp_forest,
               m_ff_options.multi_part_option, result_face. ffd_cordinates);

  //rescale final results
  for (unsigned int i = 0; i < result_face.ffd_cordinates.size(); i++) {
    result_face.ffd_cordinates[i].x /= scale;
    result_face.ffd_cordinates[i].y /= scale;
  }
};

void
FaceForest::get_paths_to_trees
  (
  std::string url,
  std::vector<std::string> &dirs
  )
{
  boost::filesystem::path dir_path(url);
  boost::filesystem::directory_iterator end_it;
  for (boost::filesystem::directory_iterator it(dir_path); it != end_it; ++it) {
    if (is_directory(it->status())) {
      dirs.push_back(it->path().string());
    }
  }
  sort(dirs.begin(), dirs.end());
  std::cout << dirs.size() << " directories found" << std::endl;
};

void
FaceForest::loading_all_trees
  (
  std::vector<std::string> urls
  )
{
  for (unsigned int i = 0; i < urls.size(); i++)
  {
    std::cout << urls[i] << std::endl;
    std::vector<Tree<MPSample>*> all_trees;
    for (int j = 0; j < num_trees; j++) {

      char buffer[200];
      sprintf(buffer, "%s/tree_%03d.txt", urls[i].c_str(), j);

      std::string tree_path = buffer;
      Forest<MPSample>::load_tree(tree_path, all_trees);
    }
    std::cout << all_trees.size() << " trees loaded" << std::endl;
    m_trees.push_back(all_trees);
  }
};
