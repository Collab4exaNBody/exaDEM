#include <omp.h>

struct cell_mutexes : public std::vector< omp_lock_t >
{
	inline void init ()
	{
		for (auto& it: *this)
			omp_init_lock(&it); 
	}
	
	inline void destroy ()
	{
		for (auto& it: *this)
			omp_destroy_lock(&it); 
	}

	inline void lock( const int i )
	{
		omp_set_lock( &this->operator[](i) );
	}

	inline void unlock( const int i )
	{
		omp_unset_lock( &this->operator[](i) );
	}
};

struct mutexes : public std::vector< cell_mutexes >
{
	mutexes() {}

	inline void initialize()
	{
#pragma omp parallel for
    for (size_t i = 0 ; i < this->size() ; i++ )
      this->operator[](i).init();
	}
	inline void destroy()
	{
#pragma omp parallel for
    for (size_t i = 0 ; i < this->size() ; i++ )
      this->operator[](i).destroy();
	}

	inline cell_mutexes& get_mutexes ( const int i )
	{
		return this->operator[](i);
	}

	inline void lock( const int cell, const int index )
	{
		this->operator[](cell).lock(index);
	}

	inline void unlock( const int cell, const int index )
	{
		this->operator[](cell).unlock(index);
	}
};
