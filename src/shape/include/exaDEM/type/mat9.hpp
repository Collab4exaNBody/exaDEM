#pragma once

/// @file
/// @brief 3 by 3 matrix
/// @author Vincent Richefeu <Vincent.Richefeu@3sr-grenoble.fr>,
/// Lab 3SR, Grenoble University

#include "vec3.hpp"

/// Matrix 3x3
template <typename T> class mat9
{
public:
  T xx, xy, xz;
  T yx, yy, yz;
  T zx, zy, zz;

  // static const mat9 zero;
  // static const mat9 unit;

  mat9() : xx(0), xy(0), xz(0), yx(0), yy(0), yz(0), zx(0), zy(0), zz(0) {}
  mat9(const T XX, const T XY, const T XZ, const T YX, const T YY, const T YZ, const T ZX, const T ZY, const T ZZ) : xx(XX), xy(XY), xz(XZ), yx(YX), yy(YY), yz(YZ), zx(ZX), zy(ZY), zz(ZZ) {}
  /*explicit*/ mat9(const T val) : xx(val), xy(val), xz(val), yx(val), yy(val), yz(val), zx(val), zy(val), zz(val) {}

  mat9(const vec3<T> &col1, const vec3<T> &col2, const vec3<T> &col3) { set(col1, col2, col3); }

  /*explicit*/ mat9(const T M[])
  {
    xx = M[0];
    xy = M[1];
    xz = M[2];
    yx = M[3];
    yy = M[4];
    yz = M[5];
    zx = M[6];
    zy = M[7];
    zz = M[8];
  }

  mat9(const mat9 &M) : xx(M.xx), xy(M.xy), xz(M.xz), yx(M.yx), yy(M.yy), yz(M.yz), zx(M.zx), zy(M.zy), zz(M.zz) {}

  mat9 &operator=(const mat9 &M)
  {
    xx = M.xx;
    xy = M.xy;
    xz = M.xz;
    yx = M.yx;
    yy = M.yy;
    yz = M.yz;
    zx = M.zx;
    zy = M.zy;
    zz = M.zz;
    return (*this);
  }

  // Constants
  static mat9 zero() { return mat9(0, 0, 0, 0, 0, 0, 0, 0, 0); }
  static mat9 unit() { return mat9(1, 0, 0, 0, 1, 0, 0, 0, 1); }
  static mat9 one() { return mat9(1, 1, 1, 1, 1, 1, 1, 1, 1); }

  void set(const vec3<T> &col1, const vec3<T> &col2, const vec3<T> &col3)
  {
    xx = col1.x;
    xy = col2.x;
    xz = col3.x;
    yx = col1.y;
    yy = col2.y;
    yz = col3.y;
    zx = col1.z;
    zy = col2.z;
    zz = col3.z;
  }

  void reset()
  {
    xx = xy = xz = 0.0;
    yx = yy = yz = 0.0;
    zx = zy = zz = 0.0;
  }

  void reset(const T val)
  {
    xx = xy = xz = val;
    yx = yy = yz = val;
    zx = zy = zz = val;
  }

  void set_diag(const T XX, const T YY, const T ZZ)
  {
    xx = XX;
    yy = YY;
    zz = ZZ;
  }

  T &operator[](int i) { return (&xx)[i]; }

  const T &operator[](int i) const { return (&xx)[i]; }

  T *c_mtx() { return &xx; }

  // Arithmetic operations
  friend mat9 operator+(const mat9 &a, const mat9 &b) { return mat9(a.xx + b.xx, a.xy + b.xy, a.xz + b.xz, a.yx + b.yx, a.yy + b.yy, a.yz + b.yz, a.zx + b.zx, a.zy + b.zy, a.zz + b.zz); }

  friend mat9 operator-(const mat9 &a, const mat9 &b) { return mat9(a.xx - b.xx, a.xy - b.xy, a.xz - b.xz, a.yx - b.yx, a.yy - b.yy, a.yz - b.yz, a.zx - b.zx, a.zy - b.zy, a.zz - b.zz); }

  friend mat9 operator-(const mat9 &a) { return mat9(-a.xx, -a.xy, -a.xz, -a.yx, -a.yy, -a.yz, -a.zx, -a.zy, -a.zz); }

  friend mat9 operator*(const mat9 &a, T k) { return mat9(k * a.xx, k * a.xy, k * a.xz, k * a.yx, k * a.yy, k * a.yz, k * a.zx, k * a.zy, k * a.zz); }

  friend mat9 operator*(const mat9 &a, const mat9 &b) { return mat9(a.xx * b.xx + a.xy * b.yx + a.xz * b.zx, a.xx * b.xy + a.xy * b.yy + a.xz * b.zy, a.xx * b.xz + a.xy * b.yz + a.xz * b.zz, a.yx * b.xx + a.yy * b.yx + a.yz * b.zx, a.yx * b.xy + a.yy * b.yy + a.yz * b.zy, a.yx * b.xz + a.yy * b.yz + a.yz * b.zz, a.zx * b.xx + a.zy * b.yx + a.zz * b.zx, a.zx * b.xy + a.zy * b.yy + a.zz * b.zy, a.zx * b.xz + a.zy * b.yz + a.zz * b.zz); }

  friend mat9 operator*(T k, const mat9 &a) { return mat9(k * a.xx, k * a.xy, k * a.xz, k * a.yx, k * a.yy, k * a.yz, k * a.zx, k * a.zy, k * a.zz); }

  friend vec3<T> operator*(const mat9 &a, const vec3<T> &v) { return vec3<T>(a.xx * v.x + a.xy * v.y + a.xz * v.z, a.yx * v.x + a.yy * v.y + a.yz * v.z, a.zx * v.x + a.zy * v.y + a.zz * v.z); }

  friend mat9 operator/(const mat9 &a, T K)
  {
    T k = 0.0;
    if (K != 0.0)
      k = 1.0 / K;
    return mat9(k * a.xx, k * a.xy, k * a.xz, k * a.yx, k * a.yy, k * a.yz, k * a.zx, k * a.zy, k * a.zz);
  }

  /// Dyadic product (tensorial product or otimes)
  // template<typename U>
  /*
           friend mat9<double> dyadic_product(const vec3<double> &a, const vec3<double> &b) {
           return mat9<double> (
           a.x * b.x, a.x * b.y, a.x * b.z,
           a.y * b.x, a.y * b.y, a.y * b.z,
           a.z * b.x, a.z * b.y, a.z * b.z
           );
           }
   */

  void operator+=(const mat9 &a)
  {
    xx += a.xx;
    xy += a.xy;
    xz += a.xz;
    yx += a.yx;
    yy += a.yy;
    yz += a.yz;
    zx += a.zx;
    zy += a.zy;
    zz += a.zz;
  }

  void operator-=(const mat9 &a)
  {
    xx -= a.xx;
    xy -= a.xy;
    xz -= a.xz;
    yx -= a.yx;
    yy -= a.yy;
    yz -= a.yz;
    zx -= a.zx;
    zy -= a.zy;
    zz -= a.zz;
  }

  void operator*=(T k)
  {
    xx *= k;
    xy *= k;
    xz *= k;
    yx *= k;
    yy *= k;
    yz *= k;
    zx *= k;
    zy *= k;
    zz *= k;
  }

  void operator/=(T K)
  {
    T k = 0.0;
    if (K != 0.0)
      k = 1.0 / K;
    xx *= k;
    xy *= k;
    xz *= k;
    yx *= k;
    yy *= k;
    yz *= k;
    zx *= k;
    zy *= k;
    zz *= k;
  }

  void setZero() { xx = xy = xz = yx = yy = yz = zx = zy = zz = 0.0; }

  void setIdentity()
  {
    xx = yy = zz = 1.0;
    xy = xz = yx = yz = zx = zy = 0.0;
  }

  T det() const { return (xx * (yy * zz - zy * yz) - yx * (xy * zz - zy * xz) + zx * (xy * yz - yy * xz)); }

  T trace() const { return (xx + yy + zz); }

  void transpose()
  {
    std::swap(xy, yx);
    std::swap(xz, zx);
    std::swap(yz, zy);
  }

  vec3<T> get_xcol() const { return vec3<T>(xx, yx, zx); }

  vec3<T> get_ycol() const { return vec3<T>(xy, yy, zy); }

  vec3<T> get_zcol() const { return vec3<T>(xz, yz, zz); }

  mat9<T> get_inverse() const
  {
    double det = xx * (yy * zz - zy * yz) - xy * (yx * zz - yz * zx) + xz * (yx * zy - yy * zx);
    double invdet;
    if (fabs(det) < 1e-20)
      invdet = 0.0; // this is a choice. Why not!
    else
      invdet = 1.0 / det;
    return mat9<T>((yy * zz - zy * yz) * invdet, -(xy * zz - xz * zy) * invdet, (xy * yz - xz * yy) * invdet,

                   -(yx * zz - yz * zx) * invdet, (xx * zz - xz * zx) * invdet, -(xx * yz - yx * xz) * invdet,

                   (yx * zy - zx * yy) * invdet, -(xx * zy - zx * xy) * invdet, (xx * yy - yx * xy) * invdet);
  }

  /// Compute eigenvectors (stored as columns in V) and corresponding eigenvalues (D)
  /// by assuming the matrix is double and symmetric
  /// See section 11.1 of Numerical Recipes in C for more information.
  int sym_eigen(mat9<T> &V, vec3<T> &D) const
  {
    int rot = 0;
    // double tresh;
    vec3<T> B;
    vec3<T> Z;

    // Save the input matrix in orig, use new matrix inp
    mat9<T> A = *this;
    // Set vectors to the identity matrix
    V.setIdentity();
    // Set B and D values to the diagonal of the input matrix
    for (uint i = 0; i < 3; i++)
      B[i] = D[i] = A[i * 3 + i];

    // Rotate until off-diagonal elements of input matrix are zero
    for (int sweep = 0; sweep++ < 50;)
    {
      double sum = fabs(A[0 * 3 + 1]) + fabs(A[0 * 3 + 2]) + fabs(A[1 * 3 + 2]);
      double thresh;

      if (fabs(sum) < 1.0e-15)
        return rot;

      thresh = (sweep < 4) ? sum * 0.2 / 9.0 : 0.0; // First three sweeps?

      for (int p = 0; p < 2; p++)
        for (int q = p + 1; q < 3; q++)
        {
          double g = 100.0 * fabs(A[p * 3 + q]);

          // After 4 sweeps, skip the rotation if the
          // off-diagonal element is small.
          if ((sweep > 4) && (g < 1.0e-15))
            A[p * 3 + q] = 0.0;
          else if (fabs(A[p * 3 + q]) > thresh)
          {
            double h = D[q] - D[p];
            double c, s, t; // cosine, sine, tangent of rotation angle
            double tau;

            if (g < 1.0e-20)
              t = A[p * 3 + q] / h;
            else
            {
              double theta = 0.5 * h / A[p * 3 + q];
              t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
              if (theta < 0.0)
                t = -t;
            }

            c = 1.0 / sqrt(1.0 + t * t); // cosine of rotation angle
            s = t * c;                   // sine of rotation angle
            tau = s / (1.0 + c);

            h = t * A[p * 3 + q];
            Z[p] -= h;
            Z[q] += h;
            D[p] -= h;
            D[q] += h;
            A[p * 3 + q] = 0.0;

            // case of rotations 0 <= j < p-1
            for (int j = 0; j <= p - 1; j++)
            {
              g = A[j * 3 + p];
              h = A[j * 3 + q];
              A[j * 3 + p] = g - s * (h + g * tau);
              A[j * 3 + q] = h + s * (g - h * tau);
            }

            // case of rotations p < j < q
            for (int j = p + 1; j < q; j++)
            {
              g = A[p * 3 + j];
              h = A[j * 3 + q];
              A[p * 3 + j] = g - s * (h - g * tau);
              A[j * 3 + q] = h + s * (g - h * tau);
            }

            // case of rotations q < j < 3
            for (int j = q + 1; j < 3; j++)
            {
              g = A[p * 3 + j];
              h = A[q * 3 + j];
              A[p * 3 + j] = g - s * (h + g * tau);
              A[q * 3 + j] = h + s * (g - h * tau);
            }

            // Set the eigen vectors
            for (int j = 0; j < 3; j++)
            {
              g = V[j * 3 + p];
              h = V[j * 3 + q];
              V[j * 3 + p] = g - s * (h + g * tau);
              V[j * 3 + q] = h + s * (g - h * tau);
            }
            rot++;
          }
        }

      // Set the eigen values
      B += Z;
      D = B;
      Z.set(0.0, 0.0, 0.0);
    }
    return -1; // Non-normal return - too many rotations
  }

  void sorted_sym_eigen(mat9<T> &V, vec3<T> &D)
  {
    this->sym_eigen(V, D);
    // sorting (descending order) bubble sort
    if (D.x < D.y)
    {
      std::swap(D.x, D.y);
      std::swap(V.xx, V.xy);
      std::swap(V.yx, V.yy);
      std::swap(V.zx, V.zy);
    }
    if (D.y < D.z)
    {
      std::swap(D.y, D.z);
      std::swap(V.xy, V.xz);
      std::swap(V.yy, V.yz);
      std::swap(V.zy, V.zz);
    }
    if (D.x < D.y)
    {
      std::swap(D.x, D.y);
      std::swap(V.xx, V.xy);
      std::swap(V.yx, V.yy);
      std::swap(V.zx, V.zy);
    }
  }

  // Comparisons
  bool operator==(const mat9<T> &other) const { return (this->xx == other.xx && this->xy == other.xy && this->xz == other.xz && this->yx == other.yx && this->yy == other.yy && this->yz == other.yz && this->zx == other.zx && this->zy == other.zy && this->zz == other.zz); }

  bool operator!=(const mat9<T> &other) const { return !(*this == other); }

  // input/output
  friend std::ostream &operator<<(std::ostream &pStr, const mat9 &M) { return (pStr << M.xx << CommBox().sep << M.xy << CommBox().sep << M.xz << CommBox().sep << M.yx << CommBox().sep << M.yy << CommBox().sep << M.yz << CommBox().sep << M.zx << CommBox().sep << M.zy << CommBox().sep << M.zz); }

  friend std::istream &operator>>(std::istream &pStr, mat9 &M) { return (pStr >> M.xx >> M.xy >> M.xz >> M.yx >> M.yy >> M.yz >> M.zx >> M.zy >> M.zz); }
};

typedef mat9<double> mat9r;
typedef mat9<float> mat9f;
typedef mat9<int> mat9i;
typedef mat9<unsigned int> mat9ui;
typedef mat9<bool> mat9b;

/// Dyadic product (tensorial product or otimes)
template <typename U> mat9<U> dyadic_product(const vec3<U> &a, const vec3<U> &b) { return mat9<U>(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z); }
