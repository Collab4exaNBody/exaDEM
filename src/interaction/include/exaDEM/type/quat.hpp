#ifndef QUAT_HPP_77D8A20E
#define QUAT_HPP_77D8A20E

/// @file
/// @brief Class for quaternions (specialized for body rotations in DEM)
/// @author Vincent Richefeu <Vincent.Richefeu@3sr-grenoble.fr>, Lab 3SR, Grenoble University
/// @date 2009-2010

#include "mat9.hpp"
#include <chrono>

/// Quaternions (spacialized class for rotations)
class quat
{
public:
	vec3r  v; ///< a vector
	double s; ///< a scalar

	quat(): v(), s(1.0) { }  // No rotation (identity) by default
	quat(double X, double Y, double Z, double S): v(X, Y, Z), s(S) { } // Take care of order
	quat(const vec3r & V, const double S): v(V), s(S) { } // Be carreful! This is NOT axis with angle of rotation!!

  quat (const quat &Q) : v(Q.v), s(Q.s) { } // copy Ctor
  
	static quat identity() { return quat(0.0, 0.0, 0.0, 1.0); }

	quat & operator = (const quat &Q) {
		v = Q.v;
		s = Q.s;
		return (*this);
	}

	quat& operator += (const quat &a) {
		s += a.s;
		v.x += a.v.x;
		v.y += a.v.y;
		v.z += a.v.z;
		return *this;
	}
	
	quat& operator -= (const quat &a) {
		s -= a.s;
		v.x -= a.v.x;
		v.y -= a.v.y;
		v.z -= a.v.z;
		return *this;
	}
	
	quat& operator *= (double k) {
		s *= k;
		v.x *= k;
		v.y *= k;
		v.z *= k;
		return *this;
	}
	
	quat& operator /= (double k) {
		s /= k;
		v.x /= k;
		v.y /= k;
		v.z /= k;
		return *this;
	}
	
	/// Product operation between two quaternions (note that q1*q2 is generally not equal to q2*q1)
	/// In term of rotations, the rotation q2 is first applied and then q1
	friend quat operator * (const quat &q1, const quat &q2) { // OP: 14*, 10±
		return quat(
			q1.v.y * q2.v.z - q1.v.z * q2.v.y + q1.s * q2.v.x + q1.v.x * q2.s,
			q1.v.z * q2.v.x - q1.v.x * q2.v.z + q1.s * q2.v.y + q1.v.y * q2.s,
			q1.v.x * q2.v.y - q1.v.y * q2.v.x + q1.s * q2.v.z + q1.v.z * q2.s,
			q1.s * q2.s - q1.v * q2.v
		);
	}

	// Rotation of a vector by a quat (Found in @see http://xrual.googlecode.com)
	// It seems to be faster than rotate method (but rotate is not deprecated since it is sure)
	vec3r operator * (const vec3r& V) const {
		// nVidia SDK implementation,
		// this is cool!!
		vec3r qvec(v.x, v.y, v.z);
		vec3r uv  = cross(qvec, V);
		vec3r uuv = cross(qvec, uv);

		uv  *= (2.0 * s);
		uuv *= 2.0;
		return (V + uv + uuv);
	}

	/// @brief Return the time derivative of the quaternion \f$ \dot{q} = \frac{1}{2} \omega q\f$
	quat dot(const vec3r &omega) {
		quat q(
		    0.5 * (omega.y * v.z - omega.z * v.y + omega.x * s),
		    0.5 * (omega.z * v.x - omega.x * v.z + omega.y * s),
		    0.5 * (omega.x * v.y - omega.y * v.x + omega.z * s),
		    -0.5 * (omega * v)
		);
		return q;
	}
	
	/// @brief Change the quaternion to its conjugate
	void conjugate() { v = -v; }
	
	quat get_conjugated() const {
		return quat(-v.x, -v.y, -v.z, s);
	}
	
	/// @brief Reset the rotation to a zero rotation
	void reset() { v.reset(); s = 1.0; }
	
	/// @brief Set the quaternion from a rotation around an axis
	/// @param[in] V      Axis (the resulting quaternion is normalized when V is normalized)
	/// @param[in] angle  Angle of rotation (radian)
	void set_axis_angle(const vec3r & V, double angle) {
		double half_angle = 0.5 * angle;
		s = cos(half_angle);
		v = sin(half_angle) * V;
	}
	
	/// @brief Set the four components of the quaternion (Be carreful! It is different from set_axis_angle)
	void set(double X, double Y, double Z, double S) {
		v.set(X, Y, Z);
		s = S;
	}
	
	/// @brief Return the rotation angle
	double get_angle() const { return (2.0 * acos(s)); }
	
	/// @brief Return Pitch angle (z-axis rotation)
	double get_Pitch() const { 
		return atan2(2.0 * (v.y * v.z + s * v.x), s * s - v.x * v.x - v.y * v.y + v.z * v.z); 
	}
	
	/// @brief Return Yaw angle (z-axis rotation)
	double get_Yaw() const { return asin(-2.0 * (v.x * v.z - s * v.y)); }
	
	/// @brief Return Roll angle (x-axis rotation)
	double get_Roll() const {
		return atan2(2.0 * (v.x * v.y + s * v.z), s * s + v.x * v.x - v.y * v.y - v.z * v.z);
	}
	
	/// @brief Return the rotation axis
	vec3r get_axis() const {
		vec3r Axis;
		double scale = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
		if (scale > 0.0) {
			scale = 1.0 / scale;
			Axis = scale * v;
		}
		return Axis;
	}

	// If V1 and V2 are not parallel, the axis of rotation is the unit-length
	// vector U = Cross(V1,V2)/Length(Cross(V1,V2)).  The angle of rotation,
	// A, is the angle between V1 and V2.  The quaternion for the rotation is
	// q = cos(A/2) + sin(A/2)*(ux*i+uy*j+uz*k) where U = (ux,uy,uz).
	//
	// (1) Rather than extract A = acos(Dot(V1,V2)), multiply by 1/2, then
	//     compute sin(A/2) and cos(A/2), we reduce the computational costs by
	//     computing the bisector B = (V1+V2)/Length(V1+V2), so cos(A/2) =
	//     Dot(V1,B).
	//
	// (2) The rotation axis is U = Cross(V1,B)/Length(Cross(V1,B)), but
	//     Length(Cross(V1,B)) = Length(V1)*Length(B)*sin(A/2) = sin(A/2), in
	//     which case sin(A/2)*(ux*i+uy*j+uz*k) = (cx*i+cy*j+cz*k) where
	//     C = Cross(V1,B).
	//
	// If V1 = V2, then B = V1, cos(A/2) = 1, and U = (0,0,0).  If V1 = -V2,
	// then B = 0.  This can happen even if V1 is approximately -V2 using
	// floating point arithmetic, since Vector3::Normalize checks for
	// closeness to zero and returns the zero vector accordingly.  The test
	// for exactly zero is usually not recommend for floating point
	// arithmetic, but the implementation of vec3::normalize guarantees
	// the comparison is robust.  In this case, the A = pi and any axis
	// perpendicular to V1 may be used as the rotation axis.
	//
	// (Found in http://xrual.googlecode.com)
	// Remark: it seems to me that V1 and V2 must be normalized (FIXME: check that)
	void set_from_to (const vec3r& V1, const vec3r& V2) {
		vec3r bisector = V1 + V2;
		bisector.normalize();

		double cos_05t = V1 * bisector;
		vec3r crossT;
		s = cos_05t;

		if (cos_05t != 0.0) {
			crossT = cross(V1, bisector);
			v.x = crossT.x;
			v.y = crossT.y;
			v.z = crossT.z;
		}
		else {
			float invLen;
			if (fabs(V1.x) >= fabs(V1.y)) {
				// V1.x or V1.z is the largest magnitude component
				invLen = 1.0 / sqrt(V1.x * V1.x + V1.z * V1.z);

				v.x = -V1.z * invLen;
				v.y = 0.0;
				v.z = +V1.x * invLen;
			}
			else {
				// V1.y or V1.z is the largest magnitude component
				invLen = 1.0 / sqrt(V1.y * V1.y + V1.z * V1.z);

				v.x = 0.0;
				v.y = +V1.z * invLen;
				v.z = -V1.y * invLen;
			}
		}
	}	
	
	/// @see http://xrual.googlecode.com
	void TwistSwingDecomp (const vec3r& V1, quat& twist, quat& swing) {
		vec3r V2 = (*this) * V1; // V2 is obtained by rotating V1 (by means of the 'this' quaternion)
		swing.set_from_to (V1, V2);
		twist = (*this) * swing.get_conjugated();
	}
	
	/// @see http://xrual.googlecode.com
	void SwingTwistDecomp (const vec3r& V1, quat& swing, quat& twist) {
		vec3r V2 = (*this) * V1;
		swing.set_from_to (V1, V2);
		twist = swing.get_conjugated() * (*this);
	}

	/// @brief Makes the quaternion normalized and return its length (before normalization of course!)
	double normalize() {
		double norm2 = s * s + v * v;
		if (norm2 <= 1e-9) {
			reset();
			return 0.0;
		}
		double norm = sqrt(norm2);
		*this *= (1.0 / norm);
		return norm;
	}
	
	/// @brief returns a random normalized quaternion
	/*
	void randomize() {
		// @see http://hub.jmonkeyengine.org/t/random-quaternions/8431
		static std::default_random_engine engine;
		static std::uniform_real_distribution<double> distrib(-1.0, 1.0);
		double sum = 0.0;
		s = distrib(engine);
		sum += s * s;
		v.x = sqrt(1 - sum) * distrib(engine);
		sum += v.x * v.x;
		v.y = sqrt(1 - sum) * distrib(engine);
		sum += v.y * v.y;
		v.z = sqrt(1 - sum) * (distrib(engine) < 0.0 ? -1.0 : 1.0);
	}
	*/
	
	void randomize(bool seedTime = false) {
		// @see http://hub.jmonkeyengine.org/t/random-quaternions/8431
		static std::default_random_engine engine;
		if (seedTime == true) engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
		static std::uniform_real_distribution<double> distrib(-1.0, 1.0);
		double sum = 0.0;
		s = distrib(engine);
		sum += s * s;
		v.x = sqrt(1 - sum) * distrib(engine);
		sum += v.x * v.x;
		v.y = sqrt(1 - sum) * distrib(engine);
		sum += v.y * v.y;
		v.z = sqrt(1 - sum) * (distrib(engine) < 0.0 ? -1.0 : 1.0);
	}
	
	// DEPRECATED!! use randomize(true) instead
	void randomizeSeed() {
		static std::default_random_engine engine;
		engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
		static std::uniform_real_distribution<double> distrib(-1.0, 1.0);
		double sum = 0.0;
		s = distrib(engine);
		sum += s * s;
		v.x = sqrt(1 - sum) * distrib(engine);
		sum += v.x * v.x;
		v.y = sqrt(1 - sum) * distrib(engine);
		sum += v.y * v.y;
		v.z = sqrt(1 - sum) * (distrib(engine) < 0.0 ? -1.0 : 1.0);
	}
	
	/// Quaternion must be normalized in order to use this function!!
	/// The builded matrix M is stored with in row-major order
	void get_rot_matrix (double M[]) const { // OP: 12*, 6+, 6-
		double Tx  = 2.0 * v.x;
		double Ty  = 2.0 * v.y;
		double Tz  = 2.0 * v.z;
		double Twx = Tx * s;
		double Twy = Ty * s;
		double Twz = Tz * s;
		double Txx = Tx * v.x;
		double Txy = Ty * v.x;
		double Txz = Tz * v.x;
		double Tyy = Ty * v.y;
		double Tyz = Tz * v.y;
		double Tzz = Tz * v.z;

		M[0] = 1.0 - (Tyy + Tzz);
		M[1] = Txy - Twz;
		M[2] = Txz + Twy;
		M[3] = Txy + Twz;
		M[4] = 1.0 - (Txx + Tzz);
		M[5] = Tyz - Twx;
		M[6] = Txz - Twy;
		M[7] = Tyz + Twx;
		M[8] = 1.0 - (Txx + Tyy);
	}
	
	/// Quaternion must be normalized in order to use this function!!
	/// The builded mat9r M is stored with in row-major order
	void get_rot_matrix (mat9r & M) const { // OP: 12*, 6+, 6-
		double Tx  = 2.0 * v.x;
		double Ty  = 2.0 * v.y;
		double Tz  = 2.0 * v.z;
		double Twx = Tx * s;
		double Twy = Ty * s;
		double Twz = Tz * s;
		double Txx = Tx * v.x;
		double Txy = Ty * v.x;
		double Txz = Tz * v.x;
		double Tyy = Ty * v.y;
		double Tyz = Tz * v.y;
		double Tzz = Tz * v.z;

		M.xx = 1.0 - (Tyy + Tzz);
		M.xy = Txy - Twz;
		M.xz = Txz + Twy;
		M.yx = Txy + Twz;
		M.yy = 1.0 - (Txx + Tzz);
		M.yz = Tyz - Twx;
		M.zx = Txz - Twy;
		M.zy = Tyz + Twx;
		M.zz = 1.0 - (Txx + Tyy);
	}
	
	// @see http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
	// remark: Q or -Q can be set
	int set_rot_matrix (double m[]) {
		double tr = m[0] + m[4] + m[8];

		if (tr > 0) {
			double S = 2.0 * sqrt(tr + 1.0); // S = 4*qw
			s = 0.25 * S;
			v.x = (m[7] - m[5]) / S;
			v.y = (m[2] - m[6]) / S;
			v.z = (m[3] - m[1]) / S;
			return 1;
		}
		else if ((m[0] > m[4]) & (m[0] > m[8])) {
			double S = sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0; // S=4*qx
			s = (m[5] - m[7]) / S;
			v.x = 0.25 * S;
			v.y = (m[1] + m[3]) / S;
			v.z = (m[2] + m[6]) / S;
			return 2;
		}
		else if (m[4] > m[8]) {
			double S = sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0; // S=4*qy
			s = (m[2] - m[6]) / S;
			v.x = (m[1] + m[3]) / S;
			v.y = 0.25 * S;
			v.z = (m[5] + m[7]) / S;
			return 3;
		}
		else {
			double S = sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0; // S=4*qz
			s = (m[3] - m[1]) / S;
			v.x = (m[2] + m[6]) / S;
			v.y = (m[5] + m[7]) / S;
			v.z = 0.25 * S;
			return 4;
		}
		return 0;
	}
	
	/// Compute \f[ P^TIP \f]
	/// where I a diagonal matrix with vector u as diagonal terms,
	/// and P the rotation matrix obtained from the quaternion (superscript T denotes for transposed)
	mat9<double> rotate_diag_mat(const vec3r& u) const {
		double P[9];
		get_rot_matrix(P);
		mat9<double> Res;

		// OP: 36*, 12+
		Res.xx =          P[0] * P[0] * u.x + P[3] * P[3] * u.y + P[6] * P[6] * u.z;
		Res.xy = Res.yx = P[0] * P[1] * u.x + P[3] * P[4] * u.y + P[6] * P[7] * u.z;
		Res.xz = Res.zx = P[0] * P[2] * u.x + P[3] * P[5] * u.y + P[6] * P[8] * u.z;
		Res.yy =          P[1] * P[1] * u.x + P[4] * P[4] * u.y + P[7] * P[7] * u.z;
		Res.yz = Res.zy = P[1] * P[2] * u.x + P[4] * P[5] * u.y + P[7] * P[8] * u.z;
		Res.zz =          P[2] * P[2] * u.x + P[5] * P[5] * u.y + P[8] * P[8] * u.z;

		return (Res);
	}
	
	/// Rotate the vector u with the rotation hold by the quaternion.
	/// Note that q*u*conj(q) require more operations (28*, 23±)
	vec3r rotate (const vec3r & u) const { // OP: 21*, 18±
		double M[9];
		get_rot_matrix(M); // OP: 12*, 12±
		vec3r v;
		// OP: 9*, 6+
		v.x = M[0] * u.x + M[1] * u.y + M[2] * u.z;
		v.y = M[3] * u.x + M[4] * u.y + M[5] * u.z;
		v.z = M[6] * u.x + M[7] * u.y + M[8] * u.z;
		return v;
	}
	
	/// Unrotate the vector u with the (oposite) rotation hold by the quaternion.
	vec3r unrotate (const vec3r & u) const {
		double M[9];
		get_rot_matrix(M);
		vec3r v;
		v.x = M[0] * u.x + M[3] * u.y + M[6] * u.z;
		v.y = M[1] * u.x + M[4] * u.y + M[7] * u.z;
		v.z = M[2] * u.x + M[5] * u.y + M[8] * u.z;
		return v;
	}
	
    bool operator == (const quat & other) const {
		return (this->s == other.s && this->v == other.v);
	}
	
	bool operator != (const quat & other) const {
		return !(*this == other);
	}
	
	// --- input/output ---
	friend std::ostream& operator<< (std::ostream& pStr, const quat& Q) {
		pStr << Q.s << CommBox().sep << Q.v;
		return pStr;
	}

	friend std::istream& operator>> (std::istream& pStr, quat& Q) {
		pStr >> Q.s >> Q.v;
		return pStr;
	}

	

}; // End class quat


namespace {
	
	// Adapted from NeHe production
	// glMultMatrixf(Matrix);
	// Note that openGL uses a column-major convention for the matrix storage
	//
	// usage: GLfloat Rot_Matrix[16];quat2GLMatrix<GLfloat>(Q, Rot_Matrix);
	template<typename floatType> 
	void quat2GLMatrix(quat & q, floatType *pMatrix)
	{
		// Make sure the matrix has allocated memory to store the rotation data
		if (!pMatrix) return;

		// First row
		pMatrix[ 0] = 1.0f - 2.0f * ( q.v.y * q.v.y + q.v.z * q.v.z );
		pMatrix[ 1] = 2.0f * (q.v.x * q.v.y + q.v.z * q.s);
		pMatrix[ 2] = 2.0f * (q.v.x * q.v.z - q.v.y * q.s);
		pMatrix[ 3] = 0.0f;

		// Second row
		pMatrix[ 4] = 2.0f * (q.v.x * q.v.y - q.v.z * q.s);
		pMatrix[ 5] = 1.0f - 2.0f * ( q.v.x * q.v.x + q.v.z * q.v.z);
		pMatrix[ 6] = 2.0f * (q.v.z * q.v.y + q.v.x * q.s);
		pMatrix[ 7] = 0.0f;

		// Third row
		pMatrix[ 8] = 2.0f * (q.v.x * q.v.z + q.v.y * q.s);
		pMatrix[ 9] = 2.0f * (q.v.y * q.v.z - q.v.x * q.s);
		pMatrix[10] = 1.0f - 2.0f * ( q.v.x * q.v.x + q.v.y * q.v.y);
		pMatrix[11] = 0.0f;

		// Fourth row
		pMatrix[12] = 0.0f;
		pMatrix[13] = 0.0f;
		pMatrix[14] = 0.0f;
		pMatrix[15] = 1.0f;

		// Now pMatrix[] is a 4x4 homogeneous · that can be applied to an OpenGL Matrix
	}
} // end namespace

#endif /* end of include guard: QUAT_HPP_77D8A20E */
