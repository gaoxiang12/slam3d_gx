#include <ctime>
#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv )
{
    srand( (unsigned int )time(0));
    for (int i=0; i<100; i++)
    {
        cout<<rand()%100<<endl;
    }
    return 0;
}
