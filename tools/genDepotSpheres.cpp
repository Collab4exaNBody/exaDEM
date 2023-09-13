#include<iostream>
#include <cmath>

template<bool print>
int build_arc(int type, double cx, double cy, double cz, double r, double n)
{
	int compt = 0;
	for(int i = 0 ; i < n ; i++)
	{
		if(i == 0)
		{
			compt++;
			if constexpr (print)
			{
				std::cout << type << " " << cx << " " << cy << " " << cz << std::endl;
			}
		}
		else
		{
			double R = ( 2 * i) * r;

			constexpr double PI = 3.14159265359;
			double angle = 180 * (2.08* r) / (PI * R) ;
			angle = angle * PI / 180;
			for( double theta = 0 ; theta+angle < 2*PI ; theta += angle)
			{
			double rx = cx + R * cos(theta);
			double ry = cy + R * sin(theta);
			double rz = cz ; 

			if constexpr (print)
			{
				compt++;
				std::cout << type << " " << rx << " " << ry << " " << rz << std::endl;
			}
			}
		}
	}
	return compt;
}


int main()
{
	const double r1 = 10;
	const double r2 = 0.5;
	const double C1 = 20;
	const double C2 = 40;
	const double C = 44;
	const double H = 100;
	const int type1 = 0;
	const int type2 = 1;
	int comp = 0;

	double epsilon = 0.01;

	const int lvl = 2;
	for(double z = r1 ; z < H ; z+= 2 * (r1 + epsilon)) 
		comp += build_arc<false>(0, C, C, z, r1, lvl);

	for(double x = 0 ; x < 2*C ; x+= 2 * (r2 + epsilon)) 
		for(double y = 0 ; y < 2*C ; y+= 2 * (r2 + epsilon)) 
			for(double z = r2 ; z < H ; z+= 2 * (r2 + epsilon)) 
			{
				const double rp = (x-C)*(x-C) + (y-C)*(y-C);
				if( rp < C2 * C2 && rp > (C1+r1+r2)*(C1+r1+r2))
				{
					comp++;
				}
			}


	std::cout << comp << std::endl;
	std::cout << 2*C << " " << 2*C << " " << H << std::endl;
	for(double z = r1 ; z < H ; z+= 2 * (r1 + epsilon)) 
		comp += build_arc<true>(0, C, C, z, r1, lvl);

	for(double x = 0 ; x < 2*C ; x+= 2 * (r2 + epsilon)) 
		for(double y = 0 ; y < 2*C ; y+= 2 * (r2 + epsilon)) 
			for(double z = r2 ; z < H ; z+= 2 * (r2 + epsilon)) 
			{
				const double rp = (x-C)*(x-C) + (y-C)*(y-C);
				if( rp < C2 * C2 && rp > (C1+r1+r2)*(C1+r1+r2))
				{
					std::cout << type2 << " " << x << " " << y << " " << z << std::endl; 
				}
			}
}
