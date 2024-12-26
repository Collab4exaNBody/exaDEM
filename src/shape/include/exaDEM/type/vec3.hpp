// Copyright (C) vec3 <vincent.richefeu@3sr-grenoble.fr>
//
// This file is part of mbox.
//
// vec3 can not be copied and/or distributed without the express
// permission of the authors.
// It is coded for academic purposes.
//
// Note
// Without a license, the code is copyrighted by default.
// People can read the code, but they have no legal right to use it.
// To use the code, you must contact the author directly and ask permission.

#pragma once

/// @file   vec3.hpp
/// @brief  Template class for vectors with 3 components
/// @author Vincent Richefeu <Vincent.Richefeu@3sr-grenoble.fr>
/// @author Lab 3SR, Grenoble University
/// @date   2009-2016

#include <iostream>
#include <cmath>
#include <random>
#include "common.hpp"

/// @brief Vector with 3 components
template <typename T> class vec3
{
public:
  T x, y, z;

  ONIKA_HOST_DEVICE_FUNC
  vec3() : x(0), y(0), z(0) {}

  ONIKA_HOST_DEVICE_FUNC
  vec3(T X, T Y, T Z) : x(X), y(Y), z(Z) {}

  ONIKA_HOST_DEVICE_FUNC
  vec3(const vec3 &v) : x(v.x), y(v.y), z(v.z) {}

  ONIKA_HOST_DEVICE_FUNC inline vec3 &operator=(const vec3 &V)
  {
    x = V.x;
    y = V.y;
    z = V.z;
    return (*this);
  }

  // Contants
  static vec3 zero() { return vec3(); }
  static vec3 unit_x() { return vec3(1, 0, 0); }
  static vec3 unit_y() { return vec3(0, 1, 0); }
  static vec3 unit_z() { return vec3(0, 0, 1); }
  static vec3 one() { return vec3(1, 1, 1); }

  ONIKA_HOST_DEVICE_FUNC inline void reset() { x = y = z = 0; }

  ONIKA_HOST_DEVICE_FUNC inline void set(T X, T Y, T Z)
  {
    x = X;
    y = Y;
    z = Z;
  }

  void set(T val) { x = y = z = val; }

  void randomize_direction(double val)
  {
    static std::default_random_engine engine;
    static std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    x = distrib(engine);
    y = distrib(engine);
    z = distrib(engine);
    normalize();
    x *= val;
    y *= val;
    z *= val;
  }

  //@LA : The value of vector components rounded to the nearest integral (as a floating-point value)
  void round(const vec3<double> &a)
  {
    if (a.x != 0.)
    {
      double x_ = x / a.x;
      x_ -= std::round(x_);
      x = x_ * a.x;
    }
    if (a.y != 0.)
    {
      double y_ = y / a.y;
      y_ -= std::round(y_);
      y = y_ * a.y;
    }
    if (a.z != 0.)
    {
      double z_ = z / a.z;
      z_ -= std::round(z_);
      z = z_ * a.z;
    }
  }

  void randomize_direction_xz(double val)
  {
    static std::default_random_engine engine;
    static std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    x = distrib(engine);
    y = 0.0;
    z = distrib(engine);
    normalize();
    x *= val;
    z *= val;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool isnull(const T tol = 1e-20) const { return (fabs(x) < tol && fabs(y) < tol && fabs(z) < tol); }

  ONIKA_HOST_DEVICE_FUNC inline T *c_vec() { return &x; }

  ONIKA_HOST_DEVICE_FUNC inline T &operator[](int i) { return (&x)[i]; }

  const T &operator[](int i) const { return (&x)[i]; }

  // For local frames, the notation n,t and s is more appropriate than x,y and z
  ONIKA_HOST_DEVICE_FUNC inline const T n() const { return x; }
  ONIKA_HOST_DEVICE_FUNC inline const T t() const { return y; }
  ONIKA_HOST_DEVICE_FUNC inline const T s() const { return z; }

  // Arithmetic operations
  ONIKA_HOST_DEVICE_FUNC inline vec3 &operator+=(const vec3 &a)
  {
    x += a.x;
    y += a.y;
    z += a.z;
    return *this;
  }

  ONIKA_HOST_DEVICE_FUNC inline vec3 &operator-=(const vec3 &a)
  {
    x -= a.x;
    y -= a.y;
    z -= a.z;
    return *this;
  }

  ONIKA_HOST_DEVICE_FUNC inline vec3 &operator*=(T k)
  {
    x *= k;
    y *= k;
    z *= k;
    return *this;
  }

  ONIKA_HOST_DEVICE_FUNC inline vec3 &operator/=(T k)
  {
    T invk = 1.0 / k;
    x *= invk;
    y *= invk;
    z *= invk;
    return *this;
  }

  ONIKA_HOST_DEVICE_FUNC inline friend vec3 operator+(const vec3 &a, const vec3 &b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }

  ONIKA_HOST_DEVICE_FUNC inline friend vec3 operator-(const vec3 &a, const vec3 &b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }

  ONIKA_HOST_DEVICE_FUNC inline friend vec3 operator-(const vec3 &a) { return vec3(-a.x, -a.y, -a.z); }

  ONIKA_HOST_DEVICE_FUNC inline friend vec3 operator*(const vec3 &a, T k) { return vec3(a.x * k, a.y * k, a.z * k); }

  ONIKA_HOST_DEVICE_FUNC inline friend vec3 operator*(T k, const vec3 &a) { return vec3(a.x * k, a.y * k, a.z * k); }

  ONIKA_HOST_DEVICE_FUNC inline friend vec3 operator/(const vec3 &a, T k)
  {
    T invk = 1.0 / k;
    return vec3(a.x * invk, a.y * invk, a.z * invk);
  }

  // --- Specific external operations ---

  /// Dot product
  ONIKA_HOST_DEVICE_FUNC inline friend T operator*(const vec3 &a, const vec3 &b) { return (a.x * b.x + a.y * b.y + a.z * b.z); }

  ONIKA_HOST_DEVICE_FUNC inline friend T dot(const vec3 &a, const vec3 &b) { return (a.x * b.x + a.y * b.y + a.z * b.z); }

  ONIKA_HOST_DEVICE_FUNC inline
      /// Multiply each component one another
      friend vec3<T>
      component_product(const vec3<T> &a, const vec3<T> &b)
  {
    return vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
  }

  ONIKA_HOST_DEVICE_FUNC inline
      /// Find the smallest components
      friend vec3<T>
      component_min(const vec3<T> &a, const vec3<T> &b)
  {
    return vec3<T>((a.x < b.x) ? a.x : b.x, (a.y < b.y) ? a.y : b.y, (a.z < b.z) ? a.z : b.z);
  }

  /// Find the biggest components
  ONIKA_HOST_DEVICE_FUNC inline friend vec3<T> component_max(const vec3<T> &a, const vec3<T> &b) { return vec3<T>((a.x > b.x) ? a.x : b.x, (a.y > b.y) ? a.y : b.y, (a.z > b.z) ? a.z : b.z); }

  /// Absolut value of the components
  ONIKA_HOST_DEVICE_FUNC inline friend vec3<T> component_abs(const vec3<T> &a) { return vec3<T>(fabs(a.x), fabs(a.y), fabs(a.z)); }

  /// Cross product
  ONIKA_HOST_DEVICE_FUNC inline friend vec3<T> operator^(const vec3<T> &a, const vec3<T> &b) { return vec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

  friend vec3<T> cross(const vec3<T> &a, const vec3<T> &b) { return vec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

  /// Squared length of the vector
  ONIKA_HOST_DEVICE_FUNC inline friend T norm2(const vec3 &a) { return a * a; }

  /// Length of the vector
  ONIKA_HOST_DEVICE_FUNC inline friend T norm(const vec3 &a) { return sqrt(a * a); }

  ONIKA_HOST_DEVICE_FUNC inline T length() const { return norm(*this); }

  /// Normalize and return length (before being normalized)
  ONIKA_HOST_DEVICE_FUNC inline T normalize()
  {
    T n = norm2(*this);
    if (n > 0.0)
    {
      n = sqrt(n);
      *this *= (1.0 / n);
    }

    return n;
  }

  ONIKA_HOST_DEVICE_FUNC inline T normalizeTested()
  {
    T n = norm2(*this);
    if (n > 0.0)
    {
      n = sqrt(n);
      *this *= (1.0 / n);
      if (x == 1)
      {
        y = z = 0;
      }
      else if (y == 1)
      {
        x = z = 0;
      }
      else if (z == 1)
      {
        x = y = 0;
      }
    }

    return n;
  }

  ONIKA_HOST_DEVICE_FUNC inline T normalizeQuotientAlgo()
  {
#   define NQ(X,Y,Z,N) \
    T f = 1.0 / Z; \
    T q1 = X * f, q2 = Y * f; \
    T h = sqrt(1.0 + q1 * q1 + q2 * q2); \
    T r = N * h; \
    Z = copysign(1.0, Z) / h; \
    X = q1 * Z; \
    Y = q2 * Z; \
    return r;

    T x1 = fabs(x), x2 = fabs(y), x3 = fabs(z);
    if (x1 > x2)
    {
      if (x3 > x1)
      {
        NQ(x, y, z, x3);
      }
      else
      {
        NQ(z, y, x, x1);
      }
    }
    else
    {
      if (x3 > x2)
      {
        if (z == 0)
          return 0;
        NQ(x, y, z, x3);
      }
      else
      {
        NQ(x, z, y, x2);
      }
    }
  }

  /// Normalize and return the normalized vector
  ONIKA_HOST_DEVICE_FUNC inline vec3 normalized()
  {
    this->normalize();
    return *this;
  }

  ONIKA_HOST_DEVICE_FUNC inline vec3 normalizedTested()
  {
    this->normalizeTested();
    return *this;
  }

  // Comparisons
  ONIKA_HOST_DEVICE_FUNC inline bool operator==(const vec3<T> &other) const { return (this->x == other.x && this->y == other.y && this->z == other.z); }

  ONIKA_HOST_DEVICE_FUNC inline bool operator!=(const vec3<T> &other) const { return !(*this == other); }

  // input/output
  friend std::ostream &operator<<(std::ostream &pStr, const vec3 &pV) { return (pStr << pV.x << CommBox().sep << pV.y << CommBox().sep << pV.z); }

  friend std::istream &operator>>(std::istream &pStr, vec3 &pV) { return (pStr >> pV.x >> pV.y >> pV.z); }
};

typedef vec3<double> vec3r;
typedef vec3<int> vec3i;
typedef vec3<unsigned int> vec3ui;
typedef vec3<bool> vec3b;

namespace std
{
  template <class T> struct less<vec3<T>>
  {
    ONIKA_HOST_DEVICE_FUNC
    inline bool operator()(const vec3<T> &lhs, const vec3<T> &rhs) const
    {
      if (rhs.x < lhs.x)
        return true;
      else if (lhs.x == rhs.x && rhs.y < lhs.y)
        return true;
      else if (lhs.x == rhs.x && lhs.y == rhs.y && rhs.z < lhs.z)
        return true;
      return false;
    }
  };
} // end namespace std
