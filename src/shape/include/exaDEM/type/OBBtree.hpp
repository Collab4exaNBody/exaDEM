// Copyright (C) <vincent.richefeu@3sr-grenoble.fr>
//
// This file is part of TOOFUS (TOols OFten USued)
//
// It can not be copied and/or distributed without the express
// permission of the authors.
// It is coded for academic purposes.
//
// Note
// Without a license, the code is copyrighted by default.
// People can read the code, but they have no legal right to use it.
// To use the code, you must contact the author directly and ask permission.

#ifndef OBB_TREE_HPP
#define OBB_TREE_HPP

#include <algorithm> // for std::sort
#include <cmath>
#include <utility> // for std::pair
#include <vector>

#include "OBB.hpp"
#include "quat.hpp"
#include "vec3.hpp"

/**
 * This is simply an OBB with some additional data
 */
template <class T> class OBBbundle {
  public:
    OBB obb;
    T data;
    std::vector<vec3r> points; // used for fitting sets of 'leafs'

    OBBbundle() {}
    OBBbundle(const OBB &obb_) : obb(obb_) {}
    OBBbundle(const OBB &obb_, T data_) : obb(obb_), data(data_) {}
};

template <class T> class OBBnode {
  public:
    OBB boundary;
    OBBnode *first = nullptr;
    OBBnode *second = nullptr;
    T data;
    OBBnode() : boundary(), first(nullptr), second(nullptr) {}
    bool isLeaf() { return (first == nullptr && second == nullptr); }
    bool isLeaf() const { return (first == nullptr && second == nullptr); }
};

template <class T> class OBBtree {
  public:
    OBBnode<T> *root = nullptr; // the root of the OBB-tree

    OBBtree() : root(nullptr) {}
    ~OBBtree() { reset(root); }

    void reset(OBBnode<T> *node) {
      if (node == nullptr)
        return;

      // first delete subtrees
      reset(node->first);
      reset(node->second);

      // then delete the node
      delete node;
      node = nullptr;
    }

    static void subdivide(std::vector<OBBbundle<T>> &OBBs, std::vector<OBBbundle<T>> &first_OBBs,
        std::vector<OBBbundle<T>> &second_OBBs) {
      if (OBBs.size() <= 1)
        return;
      if (OBBs.size() == 2) {
        first_OBBs.push_back(OBBs[0]);
        second_OBBs.push_back(OBBs[1]);
        return;
      }

      // find the axis of greatest extension
      mat9r C = OBBtree::getCovarianceMatrix(OBBs);
      mat9r eigvec;
      vec3r eigval;
      C.sorted_sym_eigen(eigvec, eigval);
      vec3r u(eigvec.xx, eigvec.yx, eigvec.zx);

      // project onto this axis
      std::vector<std::pair<double, size_t>> proj;
      for (size_t i = 0; i < OBBs.size(); i++) {
        proj.push_back(std::make_pair(OBBs[i].obb.center * u, i));
      }

      // By default the sort function sorts the vector elements on basis of first
      // element of pairs.
      std::sort(proj.begin(), proj.end());

      // distribute
      size_t half = proj.size() / 2;
      for (size_t i = 0; i < half; i++) {
        first_OBBs.push_back(OBBs[proj[i].second]);
      }
      for (size_t i = half; i < proj.size(); i++) {
        second_OBBs.push_back(OBBs[proj[i].second]);
      }
    }

    // Build the covariance matrix with the points in OBB bundles
    static mat9r getCovarianceMatrix(std::vector<OBBbundle<T>> &OBBs) {
      vec3r mu;
      mat9r C;

      // loop over the points to find the mean point
      // location
      size_t nbP = 0;
      for (size_t io = 0; io < OBBs.size(); io++) {
        for (size_t p = 0; p < OBBs[io].points.size(); p++) {
          mu += OBBs[io].points[p];
          nbP++;
        }
      }

      if (nbP == 0) {
        std::cerr << "@OBBtree::getCovarianceMatrix, no points in the OBBs!!\n";
        return C;
      }
      mu /= (double)nbP;

      // loop over the points again to build the
      // covariance matrix.  Note that we only have
      // to build terms for the upper trianglular
      // portion since the matrix is symmetric
      double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;

      for (size_t io = 0; io < OBBs.size(); io++) {
        for (size_t p = 0; p < OBBs[io].points.size(); p++) {

          vec3r pt = OBBs[io].points[p];
          cxx += pt.x * pt.x - mu.x * mu.x;
          cxy += pt.x * pt.y - mu.x * mu.y;
          cxz += pt.x * pt.z - mu.x * mu.z;
          cyy += pt.y * pt.y - mu.y * mu.y;
          cyz += pt.y * pt.z - mu.y * mu.z;
          czz += pt.z * pt.z - mu.z * mu.z;
        }
      }

      // now build the covariance matrix
      C.xx = cxx;
      C.xy = cxy;
      C.xz = cxz;
      C.yx = cxy;
      C.yy = cyy;
      C.yz = cyz;
      C.zx = cxz;
      C.zy = cyz;
      C.zz = czz;

      return C;
    }

    static OBB fitOBB(std::vector<OBBbundle<T>> &OBBs, double radius = 0.0) {
      OBB fittedObb;
      if (OBBs.empty()) {
        std::cerr << "@OBBtree::fitOBB, OBBs is empty!!\n";
        return fittedObb;
      }

      // compute the covariance matrix
      mat9r C = OBBtree::getCovarianceMatrix(OBBs);

      // ==== set the OBB parameters from the covariance matrix
      // extract the eigenvalues and eigenvectors from C
      mat9r eigvec;
      vec3r eigval;
      C.sym_eigen(eigvec, eigval);

      // find the right, up and forward vectors from the eigenvectors
      vec3r r(eigvec.xx, eigvec.yx, eigvec.zx);
      vec3r u(eigvec.xy, eigvec.yy, eigvec.zy);
      vec3r f(eigvec.xz, eigvec.yz, eigvec.zz);
      r.normalize();
      u.normalize(), f.normalize();

      // now build the bounding box extents in the rotated frame
      vec3r minim(1e20, 1e20, 1e20), maxim(-1e20, -1e20, -1e20);
      for (size_t io = 0; io < OBBs.size(); io++) {
        for (size_t p = 0; p < OBBs[io].points.size(); p++) {
          vec3r p_prime(r * OBBs[io].points[p], u * OBBs[io].points[p], f * OBBs[io].points[p]);
          if (minim.x > p_prime.x)
            minim.x = p_prime.x;
          if (minim.y > p_prime.y)
            minim.y = p_prime.y;
          if (minim.z > p_prime.z)
            minim.z = p_prime.z;
          if (maxim.x < p_prime.x)
            maxim.x = p_prime.x;
          if (maxim.y < p_prime.y)
            maxim.y = p_prime.y;
          if (maxim.z < p_prime.z)
            maxim.z = p_prime.z;
        }
      }

      // set the center of the OBB to be the average of the
      // minimum and maximum, and the extents be half of the
      // difference between the minimum and maximum
      fittedObb.center = eigvec * (0.5 * (maxim + minim));
      fittedObb.e1 = r;
      fittedObb.e2 = u;
      fittedObb.e3 = f;
      fittedObb.extent = 0.5 * (maxim - minim);

      fittedObb.enlarge(radius); // Add the Minskowski radius

      return fittedObb;
    }

    // Warning: never call the following recursive function with empty OBBs
    // Usage:
    // OBBtree obbtree;
    // obbtree.root = OBBtree::recursiveBuild(obbtree.root, OBBbundles, radius);
    static OBBnode<T> *recursiveBuild(OBBnode<T> *node, std::vector<OBBbundle<T>> &OBBs, double radius = 0.0) {
      if (OBBs.size() >= 1 && node == nullptr) {
        node = new OBBnode<T>();
        node->boundary = OBBtree<T>::fitOBB(OBBs, radius);
      } else {
        return nullptr;
      }

      if (OBBs.size() == 1) {
        node->data = OBBs[0].data;
        return node;
      }

      std::vector<OBBbundle<T>> first_subOBBs;
      std::vector<OBBbundle<T>> second_subOBBs;
      OBBtree<T>::subdivide(OBBs, first_subOBBs, second_subOBBs);

      if (!first_subOBBs.empty() && node->first == nullptr) {
        node->first = OBBtree<T>::recursiveBuild(node->first, first_subOBBs, radius);
      }
      if (!second_subOBBs.empty() && node->second == nullptr) {
        node->second = OBBtree<T>::recursiveBuild(node->second, second_subOBBs, radius);
      }
      return node;
    }

    // Usage:
    // std::vector<std::pair<myData, myData>> intersections;
    // OBBtree::TreeIntersectionIds(obbtree1.root, obbtree2.root, intersections,
    //                              scaleFactorA, scaleFactorB, enlargeValue,
    //                              posB_relativeTo_posA, QB_relativeTo_QA);
    static void TreeIntersectionIds(const OBBnode<T> *nodeA, const OBBnode<T> *nodeB, std::vector<std::pair<T, T>> &intersections,
        const double scaleFactorA, const double scaleFactorB, const double enlargeValue,
        const vec3r &posB_relativeTo_posA, const quat &QB_relativeTo_QA) {
      if (nodeA == nullptr || nodeB == nullptr) {
        return;
      }

      OBB BoundaryA = nodeA->boundary;
      BoundaryA.center *= scaleFactorA;
      BoundaryA.extent *= scaleFactorA;
      BoundaryA.enlarge(enlargeValue);

      OBB movedBoundaryB = nodeB->boundary;
      movedBoundaryB.center *= scaleFactorB;
      movedBoundaryB.extent *= scaleFactorB;
      movedBoundaryB.enlarge(enlargeValue);
      movedBoundaryB.rotate(QB_relativeTo_QA);
      movedBoundaryB.translate(posB_relativeTo_posA);

      if (BoundaryA.intersect(movedBoundaryB) == false) {
        return;
      }

      if (nodeA->isLeaf() && nodeB->isLeaf()) {
        intersections.push_back(std::pair<T, T>(nodeA->data, nodeB->data));
      } else if (!nodeA->isLeaf() && !nodeB->isLeaf()) {
        TreeIntersectionIds(nodeA->first, nodeB->first, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
        TreeIntersectionIds(nodeA->first, nodeB->second, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
        TreeIntersectionIds(nodeA->second, nodeB->first, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
        TreeIntersectionIds(nodeA->second, nodeB->second, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
      } else if (nodeA->isLeaf() && !nodeB->isLeaf()) {
        TreeIntersectionIds(nodeA, nodeB->first, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
        TreeIntersectionIds(nodeA, nodeB->second, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
      } else if (!nodeA->isLeaf() && nodeB->isLeaf()) {
        TreeIntersectionIds(nodeA->first, nodeB, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
        TreeIntersectionIds(nodeA->second, nodeB, intersections, scaleFactorA, scaleFactorB, enlargeValue,
            posB_relativeTo_posA, QB_relativeTo_QA);
      }
    }

    static void OBBIntersectionIds(const OBB &obb, const OBBnode<T> *node, std::vector<T> &intersections) {
      if (node == nullptr) {
        return;
      }

      if (obb.intersect(node->boundary) == false) {
        return;
      }

      if (node->isLeaf()) {
        intersections.push_back(node->data);
      } else if (!node->isLeaf()) {
        OBBIntersectionIds(obb, node->first, intersections);
        OBBIntersectionIds(obb, node->second, intersections);
      }
    }
};

#endif /* end of include guard: OBB_TREE_HPP */
