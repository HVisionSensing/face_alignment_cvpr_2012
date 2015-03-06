/** ****************************************************************************
 *  @file    TreeNode.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <boost/serialization/access.hpp>

/** ****************************************************************************
 * @class TreeNode
 * @brief Conditional regression tree node
 ******************************************************************************/
template<typename Sample>
class TreeNode
{
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  TreeNode
    () :
    depth(-1), right(NULL), left(NULL), is_leaf(false), has_split(false) {};

  TreeNode
    (
    int d
    ) :
    depth(d), right(NULL), left(NULL), is_leaf(false), has_split(false) {};

  ~TreeNode
    ()
  {
    if (left)
      delete left;
    if (right)
      delete right;
  };

  /*int
  getDepth
    ()
  {
    return depth;
  };*/

  bool
  isLeaf
    () const
  {
    return is_leaf;
  };

  Leaf*
  getLeaf
    ()
  {
    return &leaf;
  };

  /*void
  setLeaf
    (
    Leaf l
    )
  {
    is_leaf = true;
    leaf = l;
  };

  void
  createLeaf
    (
    const std::vector<Sample*> &samples,
    const std::vector<float> &class_weights,
    int i_leaf = -1
    )
  {
    Sample::makeLeaf(leaf, samples, class_weights, i_leaf);
    is_leaf = true;
    has_split = false;
  };

  void
  collectLeafs
    (
    std::vector<Leaf*> &leafs
    )
  {
    if (!is_leaf)
    {
      right->collectLeafs(leafs);
      left->collectLeafs(leafs);
    } else {
      leaf.depth = depth;
      leafs.push_back(&leaf);
    }
  };

  bool
  hasSplit
    () const
  {
    return has_split;
  };

  Split
  getSplit
    ()
  {
    return split;
  };

  void
  setSplit
    (
    Split s
    )
  {
    has_split = true;
    is_leaf = false;
    split = s;
  }

  void
  addLeftChild
    (
    TreeNode<Sample> *left_child
    )
  {
    left = left_child;
  };

  void
  addRightChild
    (
    TreeNode<Sample> *right_child
    )
  {
    right = right_child;
  };*/

  bool
  eval
    (
    const Sample *s
    ) const
  {
    return s->eval(split);
  };

  int depth;
  Leaf leaf;
  Split split;
  TreeNode<Sample> *right;
  TreeNode<Sample> *left;

private:
  bool is_leaf;
  bool has_split;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & depth;
    ar & is_leaf;
    ar & has_split;
    if (has_split)
      ar & split;
    if (!is_leaf)
    {
      ar & left;
      ar & right;
    }
    else
      ar & leaf;
  }
};

#endif /* TREE_NODE_HPP */
