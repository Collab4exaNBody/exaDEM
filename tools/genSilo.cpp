#include <iostream>


int main ()
{
	//	int N = 50;
	int nx = 100;
	int ny = 100;
	int nz = 2000;

	double lattice = 0.5; // metre

	double shift_x = 80.0; // should be positif
	double shift_y = 80.0;
	double shift_z = 110.0;

	int id = 0;
	int shape = 0;

	std::cout << nx*ny*nz << std::endl;
	std::cout << shift_x + (nx-1) * (2.1 * lattice) << " " << shift_y + (ny-1) * (lattice * 2.1) << " "<< shift_z + (nz-1) * (lattice * 2.1) << std::endl;

	for(int z = 0 ; z < nz ; z++)
		for(int y = 0 ; y < ny ; y++)
			for(int x = 0 ; x < nx ; x++)
			{
				std::cout 
					<< shape << " "
					<< shift_x + x * 2.1 * lattice << " "
					<< shift_y + y * 2.1 * lattice << " "
					<< shift_z + z * 2.1 * lattice << std::endl;
				id++;
			}

}

