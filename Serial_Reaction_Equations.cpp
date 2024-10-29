#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

//Note that this is a very simple serial implementation with a fixed grid and Neumann boundaries at the edges
//I am also using a vector of vectors, which is less efficient than allocating contiguous data.
vector<vector<double>> C1, C1_old, C2, C2_old;

int imax = 301, jmax = 301;
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.1, dt;
double y_max = 30.0, x_max = 30.0, dx, dy;

//set up simulation constants
const double f = 2.0, q = 0.002, epsilon = 0.03, D1 = 1.0, D2 = 0.6;

void grid_to_file(int out)
{
	//Write the output for a single time step to file
	stringstream fname1, fname2;
	fstream f1, f2;
	fname1 << "./serial_out/output_C1_" << out << ".dat";
	f1.open(fname1.str().c_str(), ios_base::out);
	fname2 << "./serial_out/output_C2_" << out << ".dat";
	f2.open(fname2.str().c_str(), ios_base::out);
	for (int i = 0; i < imax; i++)
	{
		for (int j = 0; j < jmax; j++)
		{
			f1 << C1[i][j] << "\t";
			f2 << C2[i][j] << "\t";
		}
		f1 << endl;
		f2 << endl;
	}
	f1.close();
	f2.close();
}

//Do a single time step
void do_iteration_neumann(void)
{
	//Note that I am not copying data between the grids, which would be very slow, but rather just swapping pointers
	C1.swap(C1_old);
	C2.swap(C2_old);


	//Calculate the new concentrations for all the points not on the boundary of the domain
	//Note that in parallel the edge of processor's region is not necessarily the edge of the domain
	for (int i = 1; i < imax - 1; i++)
		for (int j = 1; j < jmax - 1; j++)
		{
			C1[i][j] = C1_old[i][j] + dt * ((C1_old[i][j] * (1.0 - C1_old[i][j]) - f * C2_old[i][j] * (C1_old[i][j] - q) / (C1_old[i][j] + q)) / epsilon
				+ D1 * ((C1_old[i + 1][j] + C1_old[i - 1][j] - 2.0 * C1_old[i][j]) / (dx * dx) + (C1_old[i][j + 1] + C1_old[i][j - 1] - 2.0 * C1_old[i][j]) / (dy * dy)));

			C2[i][j] = C2_old[i][j] + dt * (C1_old[i][j] - C2_old[i][j]
				+ D2 * ((C2_old[i + 1][j] + C2_old[i - 1][j] - 2.0 * C2_old[i][j]) / (dx * dx) + (C2_old[i][j + 1] + C2_old[i][j - 1] - 2.0 * C2_old[i][j]) / (dy * dy)));
		}

	//Implement boundary conditions - This is a Neumann boundary that I have implemented
	for (int i = 0; i < imax; i++)
	{
		C1[i][0] = C1[i][1];
		C1[i][jmax - 1] = C1[i][jmax - 2];

		C2[i][0] = C2[i][1];
		C2[i][jmax - 1] = C2[i][jmax - 2];
	}

	for (int j = 0; j < jmax; j++)
	{
		C1[0][j] = C1[1][j];
		C1[imax - 1][j] = C1[imax - 2][j];

		C2[0][j] = C2[1][j];
		C2[imax - 1][j] = C2[imax - 2][j];
	}

	t += dt;
}

void do_iteration_periodic(void)
{
	//Note that I am not copying data between the grids, which would be very slow, but rather just swapping pointers
	C1.swap(C1_old);
	C2.swap(C2_old);


	//Calculate the new concentrations for all the points not on the boundary of the domain
	//Note that in parallel the edge of processor's region is not necessarily the edge of the domain
	for (int i = 0; i < imax; i++)
		for (int j = 0; j < jmax; j++)
		{
			C1[i][j] = C1_old[i][j] + dt * ((C1_old[i][j] * (1.0 - C1_old[i][j]) - f * C2_old[i][j] * (C1_old[i][j] - q) / (C1_old[i][j] + q)) / epsilon
				+ D1 * ((C1_old[(i + 1) % imax][j] + C1_old[(i - 1 + imax) % imax][j] - 2.0 * C1_old[i][j]) / (dx * dx) + (C1_old[i][(j + 1) % jmax] + C1_old[i][(j - 1 + jmax) % jmax] - 2.0 * C1_old[i][j]) / (dy * dy)));

			C2[i][j] = C2_old[i][j] + dt * (C1_old[i][j] - C2_old[i][j]
				+ D2 * ((C2_old[(i + 1) % imax][j] + C2_old[(i - 1 + imax) % imax][j] - 2.0 * C2_old[i][j]) / (dx * dx) + (C2_old[i][(j + 1) % jmax] + C2_old[i][(j - 1 + jmax) % jmax] - 2.0 * C2_old[i][j]) / (dy * dy)));
		}

	t += dt;
}

void calc_constants()
{
	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)imax - 1);

	t = 0.0;

	dt = 0.1 * pow(min(dx, dy), 2.0) / (2.0 * max(D1, D2));

	cout << "dt = " << dt << " dx = " << dx << " dy = " << dy << endl;
}

void setup_data_profile()
{
	double origin_x = x_max / 2.0, origin_y = y_max / 2.0;

	C1.resize(imax, vector<double>(jmax));
	C1_old.resize(imax, vector<double>(jmax));
	C2.resize(imax, vector<double>(jmax));
	C2_old.resize(imax, vector<double>(jmax));

	
	for (int i = 0; i < imax; i++)
		for (int j = 0; j < jmax; j++)
		{
			double x = i * dx, y = j * dy;
			double angle = atan2(y - origin_y, x - origin_x);			//Note that atan2 is not a square, but arctan taking in the x and y components separately

			if (angle > 0.0 && angle < 0.5)
				C1[i][j] = 0.8;
			else
				C1[i][j] = q * (f + 1) / (f - 1);

			C2[i][j] = q * (f + 1) / (f - 1) + angle / (8 * M_PI * f);
		}

}

int main(int argc, char *argv[])
{
	int out_cnt = 0, it = 0;

	calc_constants();
	setup_data_profile();
	grid_to_file(out_cnt);
	out_cnt++;
	t_out += dt_out;

	while (t < t_max)
	{
		//do_iteration_periodic();
		do_iteration_neumann();

		//Note that I am outputing at a fixed time interval rather than after a fixed number of time steps.
		//This means that the output time interval will be independent of the time step (and thus the resolution)
		if (t_out <= t)
		{
			cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << "\tmid " << C1[imax / 2][jmax/2] << endl;
			grid_to_file(out_cnt);
			out_cnt++;
			t_out += dt_out;
		}

		it++;
	}

	return 0;
}