/*
C++ implementation of the Nelder-Mead optimization algorithm (amoeba).
In this case algorithm search for the energetic minimum of the arrows system.
When all of the arrows are parallel, the system is in the global minimum.
GNU GPL license v3

Core algorithm code in line with
Numerical Receipes
The Art of Scientific Computing
Third Edition
W.H.Press, S.A.Teukolsky,
W.T Vetterling, B.P.Flannery

Łukasz Radziński
lukasz.radzinski _at_ gmail _dot_ com
*/

#include<cstdio>
#include<iostream>
#include<cstdlib>
#include<vector>
#include<cmath>

using namespace std;



/*
double f(vector <double> &A)
{
	//simple multidimentional parabola function for testing optimisation
	//global minimum in 0
	double s = 0.0;
	for(int i=0; i<A.size(); i++)
	{
		s += A[i]*A[i];
	}
	return s;
}*/


double f(vector <double> &A)
{
	/*
	potential energy function of the arrows system for testing optimisation
	when the neighbour arrow is parallel: E = 0
	when the neighbour arrow is orthogonal: E = 1
	when the neighbour arrow is antiparallel: E = 2
	global E is the sum of all arrows energy
	global energetic minimum is when all of the arrows are parallel, then global E = 0
	*/
	double s = 0;
	int N = A.size();
	for (int i = 0; i<N; i++)
	{
		int k = sqrt(N);
		if(i>=k) //top boundary
			s = s-cos(abs(A[i]-A[i-k]))+1;
		if(i<N-k) //bottom boundary
			s = s-cos(abs(A[i]-A[i+k]))+1;
		if(i%k!=0) //left boundary
			s = s-cos(abs(A[i]-A[i-1]))+1;
		if(i%k!=(k-1)) //right boundary
			s = s-cos(abs(A[i]-A[i+1]))+1;
	}
	return s;
}

template <class T> class Amoeba
{
	public:
		Amoeba(double func(vector <T> &v), int n)
		{
			srand (time(NULL));
			//creating point table as starting point in the system
			N = n;
			point = create_random_table();
			ndim = point.size();
			print_table(point);
			print_arrow_table(point);
			printf("Energy of the system %f\n\n", func(point));
			//creating delta values table
			vector <int> delta_tab;
			for(int i=0; i<ndim; i++)
			{
				delta_tab.push_back(delta);
			}

			//adding delta values to the point table with extended dimention as p simplex

			for(int i=0; i<ndim+1; i++)
			{
				vector<T> k;
				for(int j=0; j<ndim; j++)
				{
					k.push_back(point[j]);
				}
				p.push_back(k);
				if(i!=0)
				{
					p[i][i-1]+=delta_tab[i-1];
				}
			}
			//getting y table of solutions
			y.resize(ndim+1);
			//yhi = 0;
			mpts = p.size(); //number of rows
			for(int i = 0; i<mpts; i++)
			{
				vector <T> x;
				for(int j=0; j<ndim; j++)
				{
					x.push_back(p[i][j]);
				}
				y[i] = (func(x));
			}
			//parameters for iterating
			nfunc = 0;
			psum = get_psum(p, ndim, mpts);
			it = 0;
			int w = -1;
			while(w<0)
			{
				w = minimize(func);
			}
		};
	private:
		int N; //dimention of side of pseudo-square table
		int delta = 1; //displacement
		double ftol = 1e-6; //fractional convergence tolerance
		int NMAX = 10000000; //maximum allowed number of function evaluations
		double TINY = 1e-7; //tiny value preventing from dividing by 0
		vector <T> create_random_table();
		void print_table(vector <double> tab);
		void print_arrow(double s);
		void print_arrow_table(vector <double> tab);
		void print_result(vector <double> y, vector <vector <double> > &p, int ndim, int ilo);
		vector <double> get_psum(vector <vector <double> > &p, int ndim, int mpts);
		double amotry(vector <vector <double> > &p, vector <double> &psum, vector <double> &y, int ihi, int ndim, double fac, double func(vector <double> &v));
		int minimize(double func(vector <double> &v));
		vector<double> point;
		int ndim;
		vector <vector <double> > p;
		vector <double> y;
		//double yhi;
		int mpts;
		int it;
		int nfunc;
		vector <double> psum;
};

template <class T> vector <T> Amoeba<T>::create_random_table()
{
	/*
	creating a starting point of the system:
	n-dimentional table with random values
	values range: [0, 2*pi)
	*/
	vector <T> tab(N);
	for(int i = 0; i<N; i++)
	{
		tab[i] = fmod(rand(), 2*M_PI);
	}
	return tab;
}

template <class T> void Amoeba<T>::print_table(vector <double> tab)
{
	/*
	printing values of the table
	*/
	for(int i = 0; i<N; i++)
	{
		if(i%(int)sqrt(N)==0)
		{
			printf("\n");
		}
		printf("%+.2f\t", tab[i]);	
	}
	printf("\n");
}

template <class T> void Amoeba<T>::print_arrow(double s)
{
	/*
	printing value as an arrow with proper slope
	*/
	if(s<(M_PI/8+0*M_PI/4) || s>=(M_PI/8+7*M_PI/4))
		printf("↑ ");
	else if(s>=(M_PI/8+0*M_PI/4) && s<(M_PI/8+1*M_PI/4))
		printf("↗ ");
	else if(s>=(M_PI/8+1*M_PI/4) && s<(M_PI/8+2*M_PI/4))
		printf("→ ");
	else if(s>=(M_PI/8+2*M_PI/4) && s<(M_PI/8+3*M_PI/4))
		printf("↘ ");
	else if(s>=(M_PI/8+3*M_PI/4) && s<(M_PI/8+4*M_PI/4))
		printf("↓ ");
	else if(s>=(M_PI/8+4*M_PI/4) && s<(M_PI/8+5*M_PI/4))
		printf("↙ ");
	else if(s>=(M_PI/8+5*M_PI/4) && s<(M_PI/8+6*M_PI/4))
		printf("← ");
	else if(s>=(M_PI/8+6*M_PI/4) && s<(M_PI/8+7*M_PI/4))
		printf("↖ ");
	else
		printf("x ");
}

template <class T> void Amoeba<T>::print_arrow_table(vector <double> tab)
{
	/*
	printing values of the table as arrows with proper slope
	*/
	for(int i=0; i<N; i++)
	{
		if(i%(int)sqrt(N)==0)
			printf("\n");
		print_arrow(fmod(fmod(tab[i], 2*M_PI)+2*M_PI, 2*M_PI));
	}
	printf("\n\n");
}

template <class T> void Amoeba<T>::print_result(vector <double> y, vector <vector <double> > &p, int ndim, int ilo)
{
	/*
	printing result of optimisation
	*/
	double z = 0;
	vector <double> k = y;
	vector <vector <double> > q = p;

	z = k[0];
	k[0] = k[ilo];
	k[ilo] = z;
	vector <double> pmin(ndim);
	for(int i = 0; i<ndim; i++)
	{
		z = q[0][i];
		q[0][i] = q[ilo][i];
		q[ilo][i] = z;
		pmin[i] = q[0][i];
	}
	double fmin=k[0];
	print_table(pmin);
	print_arrow_table(pmin);
	printf("Energy of the system %f\n\n", fmin);
}

template <class T> vector <double> Amoeba<T>::get_psum(vector <vector <double> > &p, int ndim, int mpts)
{
	/*
	counting partial sum
	*/
	vector <double> psum(ndim);
	for (int j=0; j<ndim; j++)
	{
		double sum=0.0;
		for(int i=0; i<mpts; i++)
		{
			sum += p[i][j];
		}
		psum[j]=sum;
	}
	return psum;
}

template <class T> double Amoeba<T>::amotry(vector <vector <double> > &p, vector <double> &psum, vector <double> &y, int ihi, int ndim, double fac, double func(vector <double> &v))
{
	/*
	extrapolation by a factor fac through the face of the simplex across from the high point
	replacing the high point if the new point is better
	*/
	vector <double> ptry(ndim);
	double fac1=(1.0-fac)/ndim;
	double fac2=fac1-fac;
	for(int j=0; j<ndim; j++)
	{
		ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
	}
	double ytry=func(ptry);
	if (ytry < y[ihi])
	{
		y[ihi]=ytry;
		for(int j=0; j<ndim; j++)
		{
			psum[j] += ptry[j]-p[ihi][j];
			p[ihi][j]=ptry[j];
		}
	}
	return ytry;
}

template <class T> int Amoeba<T>::minimize(double func(vector <double> &v))
{
	int ilo=0;
	int ihi;
	int inhi;
	if(y[0]>y[1])
	{
		inhi = 1;
		ihi = 0;
	}
	else
	{
		inhi = 0;
		ihi = 1;
	}

	for (int i=0; i<mpts; i++)
	{
		if(y[i]<=y[ilo])
		{
			ilo = i;
		}
		if(y[i]>y[ihi])
		{
			inhi = ihi;
			ihi = i;
		}
		else if(y[i] > y[inhi] && i != ihi)
		{
			inhi = i;
		}
	}

	double rtol=2.0*abs(y[ihi]-y[ilo])/(abs(y[ihi])+abs(y[ilo])+TINY);
	if (rtol < ftol)
	{
		printf("\nIteration %d\n", it);
		print_result(y, p, ndim, ilo);
		printf("Optimisation succeeded\n");
		return 0;
	}

	if (nfunc >= NMAX)
	{
		printf("\nIteration %d\n", it);
		print_result(y, p, ndim, ilo);
		printf("Maximal number of iterations exceeded\n");
		return 1;
	}
	nfunc += 2;

	double ytry = amotry(p, psum, y, ihi, ndim, -1.0, func); //simplex reflection
	if(ytry <= y[ilo])
	{
		ytry=amotry(p, psum, y, ihi, ndim, 2.0, func); //simplex extrapolation
	}
	else if (ytry >= y[inhi])
	{
		double ysave=y[ihi];
		ytry=amotry(p, psum, y, ihi, ndim, 0.5, func); //simplex contraction
		if(ytry >= ysave)
		{
			for (int i=0;i<mpts;i++) {
				if (i != ilo) {
					for (int j=0;j<ndim;j++)
					{
						psum[j]=0.5*(p[i][j]+p[ilo][j]);
						p[i][j]=psum[j];
					}
					y[i]=func(psum);
				}
			}
			nfunc += ndim;
			psum = get_psum(p, ndim, mpts);
		}
	} else nfunc = nfunc-1;

	if(it%1000 == 0)
	{
		printf("\nIteration %d\n", it);
		print_result(y, p, ndim, ilo);
	}
	it++;
	return -1;
}

int main()
{
	Amoeba<double> a(f, 15*15);
	return 0;
}

