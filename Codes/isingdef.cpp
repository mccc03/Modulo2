#include <iostream>
#include <armadillo>
#include <iomanip>
using namespace std;
using namespace arma;


class Model
{
  public:
  int L;
  double g;
  double h;
  cx_double corr;
  cx_double corr_0;
  cx_double mu2;
  cx_double mu2_sq;
  double magx;
  double magz;
  arma:: sp_dmat *H;

  arma:: dmat *groundmat;

  //arma::vec *ground;
  //constructors
  Model(int L, double g, double h);

  //deconstructors
  ~Model();


  //class functions che specifico fuori
  void int_xx();
  void int_z_ortogonal(double g);
  void int_x_longitudinal(double h);
  void operator_magnetization(arma::dmat groundmat);
  void create_ham();
  void compute();
  void TF_corr(arma::dmat groundmat, double L);
  void mu_2(arma::dmat groundmat, double L);

  void TF_0_corr_func(arma::dmat groundmat, double L);
};


//default constructor of the Model class
// this->(variabile della classe)= parametro
Model::Model(int L, double g, double h)
{
  this->L=L;
  this->g=g;
  this->h=h;
  this->corr_0=0.0;
  this->mu2=0.0;
  this->mu2_sq=0.0;
  this->corr=0.0;
  this->H = new arma::sp_dmat(pow(2,L), pow(2,L));
  this->magz=0.0;
  this->magx=0.0;
  this->groundmat= new arma::dmat(pow(2,L),2);

}



//default deconstructors of the Model class
Model::~Model()
{
  delete this->H;
  delete this->groundmat;

  this->H= nullptr;
  this->groundmat= nullptr;
}


void Model::operator_magnetization(arma::dmat groundmat)
{
  arma:: dmat id(2,2);
  id(0,0)=1.0;
  id(0,1)=0.0;
  id(1,0)=0.0;
  id(1,1)=1.0;

  arma:: dmat sigmaz(2,2);
  sigmaz(0,1)=0.0;
  sigmaz(1,0)=0.0;        //ho def la sigma x=[0 1; 1 0]
  sigmaz(0,0)= 1.0;
  sigmaz(1,1)= -1.0;


  arma:: dmat sigmax(2,2);
  sigmax(0,1)=1.0;
  sigmax(1,0)=1.0;        //ho def la sigma x=[0 1; 1 0]
  sigmax(0,0)= 0.0;
  sigmax(1,1)= 0.0;

  arma::dmat ground = groundmat.col(0);
  arma::dmat eigvec_riga= trans(groundmat);
  arma::dmat ground_riga= eigvec_riga.row(0);

 arma::dmat provz(pow(2, L), pow(2, L));
 arma::dmat provx(pow(2, L), pow(2, L));
 for(int x=0; x<this->L; x++)
 {
   arma::dmat tempx(1,1);
   tempx(0,0)=1;
     for(int y=0; y<this->L; y++)
     {
       if(y==x)  tempx = arma::kron(tempx, sigmax);
       else tempx = arma::kron(tempx, id);
     }
     provx+= ((float)1/this->L)*tempx;
 }

  for(int x=0; x<this->L; x++)
  {
    arma::dmat tempz(1,1);
    tempz(0,0)=1;
      for(int y=0; y<this->L; y++)
      {
        if(y==x)  tempz = arma::kron(tempz, sigmaz);
        else tempz = arma::kron(tempz, id);
      }
    provz+= ((float)1/this->L)*tempz;
  }


magz= as_scalar(ground_riga*provz*ground);
magx= sqrt(as_scalar(ground_riga*provx*provx*ground));

}



//funzione della classe per int_xx
void Model::int_xx()
{
  arma:: sp_dmat id(2,2);
  id.eye(2,2);                          //ho def l'identità 2*2, si chiama id

  arma:: sp_dmat sigmax(2,2);         //ho def la sigma x=[0 1; 1 0]
  sigmax.at(0,1)= 1.0;
  sigmax.at(1,0)= 1.0;

  arma:: sp_dmat base(1,1);
  base.eye(1,1);

  arma:: sp_dmat temp(2,2);
  for(int x=0; x<this-> L-1; x++)
  {
    temp= base;
    for(int y=0; y<this->L; y++)
    {
      if(y==x || y==x+1)  temp= arma::kron(temp, sigmax);
      else temp= arma::kron(temp, id);
    }
    //add to H
    (*this->H)= (*this->H)-temp;
  }

  //termine collegato a PBC (chiusura anello)
  temp = arma::kron(base, sigmax);
    for(int x=1; x<this->L-1; x++) temp = arma::kron(temp, id);
    temp = arma::kron(temp, sigmax);
    (*this->H) = (*this->H) - temp; // add to hamiltonian

}



//funzione della classe per int ortogonale
void Model:: int_z_ortogonal(double g)
{
  arma:: sp_dmat id(2,2);
  id.eye(2,2);                          //ho def l'identità 2*2, si chiama id

  arma:: sp_dmat sigmaz(2,2);         //ho def la sigma z=[1 0; 0 -1]
  sigmaz.at(0,0)= 1.0;
  sigmaz.at(1,1)= -1.0;

  arma:: sp_dmat base(1,1);
  base.eye(1,1);

  arma:: sp_dmat temp(2,2);
  for(int x=0; x<this-> L; x++)
  {
    temp=base;
      for(int y=0; y<this->L; y++)
      {
        if(y==x)  temp = arma::kron(temp, sigmaz);
        else temp = arma::kron(temp, id);
      }
  (*this->H)= (*this->H)- g* temp;
  }

}



void Model:: int_x_longitudinal(double h)
{
  arma:: sp_dmat id(2,2);
  id.eye(2,2);                          //ho def l'identità 2*2, si chiama id

  arma:: sp_dmat sigmax(2,2);         //ho def la sigma x=[0 1; 1 0]
  sigmax.at(0,1)= 1.0;
  sigmax.at(1,0)= 1.0;

  arma:: sp_dmat base(1,1);
  base.eye(1,1);

  arma:: sp_dmat temp(2,2);
  for(int x=0; x<this-> L; x++)
  {
    temp=base;
      for(int y=0; y<this->L; y++)
      {
        if(y==x)  temp = arma::kron(temp, sigmax);
        else temp = arma::kron(temp, id);
      }
  (*this->H) = (*this->H) - h* temp;
  }

}



void Model:: TF_corr(arma::dmat groundmat, double L)
{
const double pi = std::acos(-1);
const std::complex<double> l(0, 1);
const double due=2.0;
arma::dmat op(2,2);

arma::dmat id(2,2);
id(0,0)=1.0;
id(0,1)=0.0;
id(1,0)=0.0;
id(1,1)=1.0;

arma::dmat sigmax(2,2);
sigmax(0,1)=1.0;
sigmax(0,0)=0.0;
sigmax(1,0)=1.0;
sigmax(1,1)=0.0;

  arma::dmat ground = groundmat.col(0);
  arma::dmat eigvec_riga= trans(groundmat);
  arma::dmat ground_riga= eigvec_riga.row(0);     //così ho sia ground riga che ground colonna

    for(double i=0; i<L; i++)
      {
        for(double j=0; j<L; j++)
            {
              arma::dmat temp(1,1);
              temp(0,0)=1;
              if(i==j) op=id;
              else op=sigmax;
                for(int y=0; y<L ; y++)
                {
                  if(y==i || y==j) temp = arma::kron(temp, op);
                  else temp= arma::kron(temp,id);
                }
                arma::cx_double cortemp=as_scalar(ground_riga*temp*ground);
                corr = corr + exp(l*due*pi*(i-j)/L)*cortemp;

             }
      }
}

void Model::mu_2(arma::dmat groundmat, double L)
{

  arma::dmat ground = groundmat.col(0);
  arma::dmat eigvec_riga= trans(groundmat);
  arma::dmat ground_riga= eigvec_riga.row(0);

  arma::dmat prov(pow(2, L), pow(2, L));
    // define trivial matrices
    arma::dmat id(2,2);
    id(0,0)=1.0;
    id(0,1)=0.0;
    id(1,0)=0.0;
    id(1,1)=1.0;

    arma::dmat sigmax(2,2);
    sigmax(0,1)=1.0;
    sigmax(0,0)=0.0;
    sigmax(1,0)=1.0;
    sigmax(1,1)=0.0;

//costruisco (sum_i sigma_i^x)^2 x capire se coincide con G(p) dopo che ne ho fatto valor medio

    for(int i=0; i<L; i++)
       {
          arma::dmat temp(1,1);
          temp(0,0)=1;
           for(int y=0; y<L; y++)
           {
               if(y == i) temp = arma::kron(temp, sigmax);
               else temp = arma::kron(temp, id);
           }
          prov+= temp;
       }
       arma::dmat prov2= prov*prov;     //così ho oggetto (sum_i sigma_i)^2
       mu2=as_scalar(ground_riga*prov2*ground);  //così ho <mu2>L^2
       mu2= mu2/pow(L,2);
       mu2= pow(mu2,2);             //ho <mu2>

       arma::dmat prov3= prov*prov*prov*prov;   //(sum_i sigma_i)^4  voglio <mu2^2>
       mu2_sq=as_scalar(ground_riga*prov3*ground);  //così ho <mu2>L^4
       mu2_sq= mu2_sq/pow(L,4);

}



void Model::TF_0_corr_func(arma::dmat groundmat, double L)
{
    arma::dmat id(2,2);
    id(0,0)=1.0;
    id(0,1)=0.0;
    id(1,0)=0.0;
    id(1,1)=1.0;

    arma::dmat sigmax(2,2);
    sigmax(0,1)=1.0;
    sigmax(0,0)=0.0;
    sigmax(1,0)=1.0;
    sigmax(1,1)=0.0;

    //arma::dmat base=id;

  const double pi = std::acos(-1);
  const std::complex<double> l(0, 1);

    // define trivial matrices
    const double due=2.0;
    arma::dmat op(2,2);
    arma::dmat ground = groundmat.col(0);
    arma::dmat eigvec_riga= trans(groundmat);
    arma::dmat ground_riga= eigvec_riga.row(0);     //così ho sia ground riga che ground colonna

            for(double i=0; i<L; i++)
            {
              for(double j=0; j<L; j++)
              {
                arma::dmat temp(1,1);
                temp(0,0)=1;
                if(i==j) op=id;
                else op=sigmax;
                for(int y=0; y<L ; y++)
                {
                    if(y==i || y==j) temp = arma::kron(temp, op);
                    else temp= arma::kron(temp,id);
                }
              corr_0+= as_scalar(ground_riga*temp*ground);

            }
          }
}



void Model::compute()
{
    arma::vec eigval;
    arma::dmat eigvec;

    arma::eigs_sym(eigval, eigvec, (*this->H), 3, "sa");
    (*this->groundmat)= eigvec;
    //arma::vec real_ground = eigvec.col(0);
    //for(int x=0; x<real_ground.size(); x++) this->ground->at(x) = real_ground.at(x);
    this->operator_magnetization(*this->groundmat);

  //this-> TF_corr((*this->groundmat),this->L);
  //this->mu_2((*this->groundmat), this->L);
   //this->TF_0_corr_func((*this->groundmat), this->L);

}


void Model:: create_ham()
{
  this->int_xx();
  this->int_z_ortogonal(this->g);
  this->int_x_longitudinal(this->h);

}



class qt_sys     //una classe per i modelli bipartiti dtotale=da*db
  {
  public:
    double fid;

    arma::dmat groundmat1;
    arma::dmat groundmat2;
  //constructors
    qt_sys( arma::dmat groundmat1, arma::dmat groundmat2 );

    //deconstructors
    ~qt_sys();


    //class functions che specifico fuori

    void fidelity(arma::dmat groundmat1 , arma::dmat groundmat2);
    void create();
  };


    //default constructor of the Model class
    // this->(variabile della classe)= parametro
    qt_sys::qt_sys(arma::dmat groundmat1,arma::dmat groundmat2)
    {

      this->fid=0.0;

      this->groundmat1= groundmat1;
      this->groundmat2=groundmat2;

    }


    //default deconstructors of the Model class
    qt_sys::~qt_sys()
    {
      /*
      delete this->rho;
      delete this->rho_A;
    //  delete this->rho_B;

      this->rho=nullptr;
      this->rho_A= nullptr;
    //  this-> rho_B= nullptr;
    */
    }

void qt_sys::fidelity(arma::dmat groundmat1,arma::dmat groundmat2)
{

    // define trivial matrices
    arma::dmat ground1 = groundmat1.col(0);
    arma::dmat eigvec_riga1= trans(groundmat1);
    arma::dmat ground_riga1= eigvec_riga1.row(0);

    arma::dmat ground2 = groundmat2.col(0);
    arma::dmat eigvec_riga2= trans(groundmat2);
    arma::dmat ground_riga2= eigvec_riga2.row(0);

    fid= as_scalar(ground_riga1*ground2);
}


  void qt_sys::create()
  {
// this->Dens_Matrix_create(this->groundmat);
 //this->TF_corr_func(this->groundmat1);
 //this->mu_2(this->groundmat);
 this->fidelity(this->groundmat1, this->groundmat2);
 //this->susc(this->groundmat);
 //this->TF_0_corr_func(this->groundmat1);
  }


int main(int argc, char **argv)
{
  const double pi= acos(-1);
  const int n =16;

  Model mymodel1(4, 1.0, 0.0);
//  Model mymodel2(8, 0.8, 0);
  //Model mymodel3(8, 1.0, 0);




  //creating the hamiltonian matrix
  //mymodel1.create_ham();
  //mymodel2.create_ham();
  //mymodel3.create_ham();
/*
  arma:: vec eigval1;
  arma:: mat eigvec1;

  arma:: vec eigval2;
  arma:: mat eigvec2;

  arma:: vec eigval3;
 arma:: mat eigvec3;

  arma::eigs_sym(eigval1, eigvec1, (*mymodel1.H), 3, "sa");
 arma::eigs_sym(eigval2, eigvec2, (*mymodel2.H), 3, "sa");
  arma::eigs_sym(eigval3, eigvec3, (*mymodel3.H), 3,"sa");

  arma::dmat ground = eigvec1.col(0);
  arma::dmat eigvec_riga= trans(ground);
  arma::dmat ground_riga= eigvec_riga.row(0);

  arma::dmat ground2 = eigvec2.col(0);
  arma::dmat eigvec_riga2= trans(ground2);
  arma::dmat ground_riga2= eigvec_riga2.row(0);

  arma::dmat ground3 = eigvec3.col(0);
  arma::dmat eigvec_riga3= trans(ground3);
  arma::dmat ground_riga3= eigvec_riga3.row(0);

cout << ground_riga << endl;
cout << ground_riga2 << endl;
cout << ground_riga3 << endl;
*/
  //std::cout  << "ground e first excited per g=1,h=0.8 è \n " << eigval1[0] << "\n" << eigval1[1] << std::endl;
//  std::cout  << "ground e first excited per g=0.2 è \n " << eigval2[0] << "\n" << eigval2[1] << std::endl;
 //std::cout  << "gap è:  " << eigval1[1]-eigval1[0]  << std::endl;
  //std::cout << eigval2[0] << std::endl;
/*
mymodel1.compute();
arma::cx_double corr= mymodel1.corr;
arma::cx_double corr_0=mymodel1.corr_0;
cx_double diff= (corr_0-corr);
diff= diff/ (4*pow(sin(pi/11),2));
diff= diff/corr;
diff= sqrt(diff.real());
double corrlen= diff.real();
cout << corrlen << endl;
*/


/*
    int NUM=21;
    double *g= new double[NUM];
    double *corrlen= new double[NUM];

   for (double L=6; L<11; L++)
   {

      for(int i=0; i<NUM; i++)
      {
        for(double j=-0.1 ; j<2.0 ; i++)
        {
          j=j+0.1;
          //u[i]=j;
          g[i]= j;
          Model modelxplot(L, g[i] , 0.0);
           modelxplot.create_ham();
           modelxplot.compute();
           cx_double corr_plot= modelxplot.corr;
           cx_double corr_0_plot= modelxplot.corr_0;
           cx_double diff= (corr_0_plot-corr_plot);
           diff= diff/ (4*pow(sin(pi/L),2));
           diff= diff/corr_plot;
           diff= sqrt(diff.real());
           corrlen[i]= diff.real();
          // cout << corrlen[i] << endl;
         }
       }

      fstream myfile;
      myfile.open ("provola3.txt",  fstream::out | fstream::app);

      for(int i = 0; i < NUM; i++)
      {
        myfile  <<  g[i] << " " <<  std::setprecision(16) << corrlen[i]/L << std::endl;
      }
    }
*/

/*
int NUM=21;
double *h= new double[NUM];
double *fid= new double[NUM];

for (double L=8; L<11; L++)
{

  double m_0= pow((1- pow(atof(argv[1]),2)),1/8);
  double delta_f= 2*sqrt((1-pow(atof(argv[1]),2))/(pi*L))*pow(atof(argv[1]),L);


  for(int i=0; i<NUM; i++)
  {
    for(double j=-1.0 ; j<1.1; i++)
    {
      j=j+0.1;
      //u[i]=j;
      h[i]= j;


      Model modelxplot1(L ,atof(argv[1]) , h[i]*delta_f/(2*m_0*L) );     //L,g,h
      Model modelxplot2(L , atof(argv[1]) ,  (0.0001+h[i])*delta_f/(2*m_0*L) );     //L,g,h

      modelxplot1.create_ham();
      modelxplot1.compute();

      modelxplot2.create_ham();
      modelxplot2.compute();


      qt_sys qtxplot( (*modelxplot1.groundmat) , (*modelxplot2.groundmat) );

       qtxplot.create();

       fid[i]= qtxplot.fid;
      // cout << fid[i] << endl;
     }
   }

  fstream myfile;
  myfile.open ("provola3.txt",  fstream::out | fstream::app);

  for(int i = 0; i < NUM; i++)
  {
    myfile  <<  h[i]<< " " <<  std::setprecision(16) << fid[i]<< std::endl;
  }
}
*/

/*
int NUM=21;
double *g= new double[NUM];
cx_double *U= new cx_double[NUM];
//  int *L= new int[8];

for (int L=6; L<11; L++)
{
  for(int i=0; i<NUM; i++)
  {
    for(double j=-0.1 ; j<2.0 ; i++)
    {
      j=j+0.1;
      //u[i]=j;
      g[i]= j;
      Model modelxplot(L , g[i], 0.0);
      modelxplot.create_ham();
      modelxplot.compute();
      cx_double mu2_first= modelxplot.mu2;
      cx_double mu2_second= modelxplot.mu2_sq;
      U[i]=mu2_second/mu2_first;

    }
  }
  fstream myfile;
  myfile.open ("provola2.txt",  fstream::out | fstream::app);

  for(int i = 0; i < NUM; ++i)
  {
    myfile  <<  g[i] << " " <<  std::setprecision(16) << U[i]  << std::endl;
  }
}
*/


int NUM=21;
double *g= new double[NUM];
//double *magz= new double[NUM];
double *magx= new double[NUM];
//double *magx1= new double[NUM];


for (double L=4; L<5; L++)
{
  //double m_0=pow((1-pow(atof(argv[1]),2)),0.125);
  //double delta= 2*sqrt((1-pow(atof(argv[1]),2))/(pi*L))*pow(atof(argv[1]),L);
  for(int i=0; i<NUM; i++)
  {
    for(double j=-0.1 ; j<2.0; i++)
    {
      j=j+0.1;
      //u[i]=j;
      g[i]= j;
      Model modelxplot(L, g[i] , 0.0);
      //Model modelxplot1(L, 1.0 , g[i]-0.0005);
       modelxplot.create_ham();
       modelxplot.compute();

       //modelxplot1.create_ham();
       //modelxplot1.compute();

       //magx1[i]= modelxplot1.magx;
       magx[i]= modelxplot.magz;

       //magx1[i]= modelxplot1.magx;
      // cout << corrlen[i] << endl;
     }
   }

  fstream myfile;
  myfile.open ("provola3.txt",  fstream::out | fstream::app);

  for(int i = 0; i < NUM; i++)
  {
    myfile  <<  g[i] << " " <<  std::setprecision(16) << magx[i] << std::endl;
  }
}


//GAP IN FUNZIONE DI L
/*
  int NUM=21;
  double *h= new double[NUM];
  double *ground_energy = new double[NUM];
  double *first_excited = new double[NUM];
  double *gap           = new double[NUM];

for (int L=2; L<13; L++)
{
  //double m_0=pow((1-pow(atof(argv[1]),2)),0.125);
  //double delta= 2*sqrt((1-pow(atof(argv[1]),2))/(pi*L))*pow(atof(argv[1]),L);

for(int i=0; i<NUM; i++)
{
  for(double j=-0.1; j<2.0; i++)
   {
       j=j+0.1;
       h[i]=j;
      Model modelxplot(L ,h[i] , 0.0);     //L,g,h
      modelxplot.create_ham();
      arma:: vec  temp_eigval;
      arma:: dmat temp_eigvec;
      //arma:: vec  temp_gap;
      arma::eigs_sym(temp_eigval, temp_eigvec, (*modelxplot.H), 3, "sa");
      ground_energy[i]= temp_eigval[0];
      first_excited[i]= temp_eigval[1];
      gap[i]= first_excited[i]-ground_energy[i];


      //gap[i]          = first_excited[i]-ground_energy[i];
    }
  }

    fstream myfile;
    myfile.open ("ising.txt",  fstream::out | fstream::app);
    for (int i=0; i<NUM; i++)
    {
      myfile << std::setprecision(16) << h[i] << " " << gap[i] << std::endl;
    }
}

//myfile.close();
*/


/*
double magx;
Model modelxplot(10, atof(argv[1]), -0.5);
 modelxplot.create_ham();
 modelxplot.compute();
 //magz[i]= modelxplot.magz;
 magx= modelxplot.magx;
 cout << "mx is" << magx << endl;
*/

  return EXIT_SUCCESS;


}
