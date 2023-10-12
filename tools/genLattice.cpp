#include <iostream>


int main ()
{
	//	int N = 50;
	int nx = 10;
	int ny = 10;
	int nz = 10;

	double lattice = 0.5; // metre

	int id = 0;
	int shape = 0;

	std::cout << nx*ny*nz << std::endl;
	std::cout << (nx-1) * (2.1 * lattice) << " " << (ny-1) * (lattice * 2.1) << " "<< (nz-1) * (lattice * 2.1) << std::endl;

	for(int z = 0 ; z < nz ; z++)
		for(int y = 0 ; y < ny ; y++)
			for(int x = 0 ; x < nx ; x++)
			{
				std::cout 
					<< shape << " "
					<< x * 2.1 * lattice << " "
					<< y * 2.1 * lattice << " "
					<< z * 2.1 * lattice << std::endl;
				id++;
			}

}

