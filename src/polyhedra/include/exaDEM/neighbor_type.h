#pragma once

#include <exanb/core/basic_types.h>
#include <onika/memory/allocator.h> // for DEFAULT_ALIGNMENT
#include <onika/cuda/cuda.h> 
#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS constant
#include <exanb/io/cell_array_serializer.h>

#include <exaDEM/neighbor_type_base.h>

#include <cstring>
#include <iomanip>

namespace exaDEM
{
  using namespace exanb;

  /************************************************************************************
   * Storage for neighborhood type for all items in a cell.
   * a single array is used to ease allocation, GPU access and Communication of data.
   ***********************************************************************************/
	template<typename T>
  struct CellParticleNeighborT
  {
    using Word = uint64_t;
    static_assert( alignof(T) % sizeof(Word) == 0 );
    static constexpr size_t PairTWords = sizeof(T) /  sizeof(Word) ; 
    
    onika::memory::CudaMMVector<Word> m_data;  

    inline void initialize( size_t n_items )
    {
      if( n_items == 0 )
      {
        m_data.clear();
        return;
      }
      const size_t index_count = n_items + 2;
      m_data.assign(index_count,0);
      m_data[0] = n_items; // number of items
      m_data[1] = 0; // total number of pair types
    }

    inline const uint8_t* read_from_stream( const uint8_t* stream )
    {
      const Word* streamw = ( const Word * ) stream;
      const size_t n_items = streamw[0];
      const size_t total_pairs = streamw[1];
      const size_t payload = n_items+2 + n_items*2 + total_pairs * PairTWords ;
      m_data.assign( streamw , streamw + payload );
      return ( const uint8_t* ) ( streamw + payload );
    }

    inline void clear()
    {
      m_data.clear();
    }

    inline size_t number_of_items() const
    {
      if( m_data.empty() ) return 0;
      else return m_data[0];
    }

    inline bool check_consistency()
    {
      if( m_data.empty() )
      {
        return true;
      }
      else
      {
        assert( m_data.size() >= 2 );
        const size_t n_items = m_data[0];
        assert( m_data.size() >= 2 + n_items + n_items*2 );
        const size_t total_pairs = m_data[1];
        const size_t payload = n_items+2 + n_items*2 + total_pairs * PairTWords ;
        size_t pfcount = 0;
        for(size_t p=0;p<n_items;p++) pfcount += m_data[ m_data[2+p]-1 ];
        return m_data.size()>=2 && m_data.size() == payload && total_pairs == pfcount ;
      }
    }

    // remove null type pairs
    inline std::pair<size_t,size_t> compact_type_pairs()
    {
      if( m_data.empty() ) return {0,0};
      
      const size_t total_pairs_before = m_data[1];
      //m_data[1] = 0;
      size_t total_pairs = 0;
      size_t data_pos = m_data[0] + 2;
      const size_t n_items = number_of_items();
      for(size_t p=0;p<n_items;p++)
      {
        Word in_data_pos = m_data[2+p];
        Word id = m_data[ in_data_pos - 2 ];
        Word n_in_pairs = m_data[ in_data_pos - 1 ];
        
        m_data[ data_pos ] = id;
        m_data[ data_pos + 1 ] = 0; // updated later on
        m_data[2+p] = data_pos + 2;
        
        const ParticlePairT<T> * in_ppf_base = ( const ParticlePairT<T> * ) ( m_data.data() + in_data_pos );
        ParticlePairT<T> * out_ppf_base = ( ParticlePairT<T> * ) ( m_data.data() + data_pos + 2 );
        
        size_t n_out_pairs = 0;
        for(size_t f=0;f<n_in_pairs;f++)
        {
          if( ! in_ppf_base[f].is_null() )
          {
            if( in_ppf_base+f != out_ppf_base+n_out_pairs ) out_ppf_base[n_out_pairs] = in_ppf_base[f];
            ++ n_out_pairs;
            ++ total_pairs;
          }
        }
        m_data[ data_pos + 1 ] = n_out_pairs;
        data_pos += 2 + n_out_pairs * PairTWords;
        assert( data_pos <= m_data.size() );
      }
      m_data[1] = total_pairs;
      m_data.resize( data_pos );
      assert( check_consistency() );
      return { total_pairs_before , total_pairs };
    }

    inline void begin_nbh_write(size_t p, uint64_t id) // prepare for inserting ParticlePairT elements for particle p
    {
      m_data.push_back( id ); // id of particle
      m_data.push_back( 0 ); // number of type pairs for this particle
      m_data[2+p] = m_data.size();
    }

    inline void push_back( const ParticlePairT<T>& ppf )
    {
      const size_t payload_before = m_data.size();
      ++ m_data[1];
      m_data.resize( payload_before + PairTWords );
      * ( (ParticlePairT<T>*) & m_data[payload_before] ) = ppf;
    }

    inline void end_nbh_write(size_t p, uint64_t id) // prepare for inserting ParticlePairT elements for particle p
    {
      assert( m_data[ m_data[2+p] - 1 ] == 0 );
      assert( m_data[ m_data[2+p] - 2 ] == id );
      const size_t inserted_words = m_data.size() - m_data[2+p];
      assert( inserted_words % PairTWords == 0 );
      const size_t inserted_pairs = inserted_words / PairTWords;
      m_data[ m_data[2+p] - 1 ] = inserted_pairs;
    }

    inline Word particle_id( size_t p ) const
    {
      return m_data[ m_data[2+p] - 2 ];
    }

    inline Word particle_number_of_pairs( size_t p ) const
    {
      return m_data[ m_data[2+p] - 1 ];
    }

    inline std::pair< const Word *, const Word * > particle_data_range(size_t p) const
    {
      const auto * start = & m_data[ m_data[2+p] - 2 ];
      const auto * end = start + particle_number_of_pairs(p) * PairTWords + 2 ;
      return { start , end };
    }

    // append data for items [pstart;pend[ in raw stream dataw
    inline void append_data_stream_range( const Word * dataw, size_t pstart, size_t pend )
    {
      if( pstart==0 && pend==dataw[0] && number_of_items()==0 ) // fast path
      {
        const size_t payload = 2 + dataw[0] + dataw[0]*2 + dataw[1] * PairTWords ;
        m_data.assign( dataw , dataw+payload );
        //lout<<"* append : FAST"<<std::endl;
      }
      else
      {
        //lout<<"* append : CONCAT"<<std::endl;
      
        assert( pstart < pend );
        const size_t n_items = number_of_items();
        const size_t n_append_items = pend - pstart;

        CellParticleNeighborT<T> concat;
        concat.initialize( n_items + n_append_items );
        
        for(size_t p=0;p<n_items;p++)
        {
          concat.begin_nbh_write( p , particle_id(p) );
          const size_t n_pairs = particle_number_of_pairs( p );
          for(size_t i=0;i<n_pairs;i++)
          {
            concat.push_back( pair_type(p,i) );
          }
          concat.end_nbh_write( p , particle_id(p) );
        }
        for(size_t p=pstart;p<pend;p++)
        {
          const size_t pi = n_items + p - pstart;
          const size_t offset = dataw[2+p];
          assert( offset > 2 );
          const uint64_t id = dataw[ offset - 2 ];
          const size_t n_pairs = dataw[ offset - 1 ];
          const ParticlePairT<T> * pf_base = ( const ParticlePairT<T> * ) ( dataw + offset );
          //lout<<std::setprecision(5);
          //lout<<"import type: id="<<id<<", n_pairs="<<n_pairs<<std::endl;
          concat.begin_nbh_write( pi , id );
          for(size_t i=0;i<n_pairs;i++)
          {
            //lout<<"\t{" << pf_base[i].m_particle_id << ",(" << pf_base[i].m_type.x<<"," << pf_base[i].m_type.y<<","<<pf_base[i].m_type.z<<")}" << std::endl;
            concat.push_back( pf_base[i] );
          }
          concat.end_nbh_write( pi , id );
        }
        m_data = std::move( concat.m_data );
        assert( check_consistency() );
      }
    }

    inline void shrink_to_fit()
    {
      m_data.shrink_to_fit();
    }

    inline size_t storage_size() const
    {
      return m_data.size() * sizeof(Word);
    }

    inline void set_storage_size(size_t sz)
    {
      assert( sz % sizeof(Word) == 0 );
      assert( sz / sizeof(Word) >= 2 || sz==0 );
      //m_data.clear();
      m_data.assign( sz / sizeof(Word) , 0 );
    }

    inline const uint8_t* storage_ptr() const
    {
      return (const uint8_t*) m_data.data();
    }

    inline uint8_t* storage_ptr()
    {
      return (uint8_t*) m_data.data();
    }
        
    ONIKA_HOST_DEVICE_FUNC inline const ParticlePairT<T>& pair_type(unsigned int p, unsigned int n) const
    {
      const Word * __restrict__ data_ptr = onika::cuda::vector_data( m_data );
      const ParticlePairT<T> * __restrict__ ppf_base = ( const ParticlePairT<T> * ) ( data_ptr + data_ptr[2+p] );
      return ppf_base[ n ];
    }

    ONIKA_HOST_DEVICE_FUNC inline ParticlePairT<T>& pair_type(unsigned int p, unsigned int n)
    {
      Word * __restrict__ data_ptr = onika::cuda::vector_data( m_data );
      ParticlePairT<T> * __restrict__ ppf_base = ( ParticlePairT<T> * ) ( data_ptr + data_ptr[2+p] );
      return ppf_base[ n ];
    }
  };



  /************************************************************************************
   * neighborhod type storage for the whole grid.
   ***********************************************************************************/
	template<typename T>
  struct GridCellParticleNeigborT
  {
    onika::memory::CudaMMVector< CellParticleNeighborT<T> > m_cell_type;
  };


  // ====================================================================================
  // ================ compute_pair_singlemat interoperability utilities =================
  // ====================================================================================

  /************************************************************************************
   * Compute buffer extension for neighbor type
   ***********************************************************************************/
  template<typename T, bool _HasT , size_t _MaxNeighbors=exanb::MAX_PARTICLE_NEIGHBORS>
  struct ParticleNeighborTBuffer
  {
    static inline constexpr bool c_has_nbh_data = _HasT;
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;

    alignas(onika::memory::DEFAULT_ALIGNMENT) ParticlePairT<T> m_data[MaxNeighbors];  

    ONIKA_HOST_DEVICE_FUNC void set( size_t nbh_index, const ParticlePairT<T>& ppf ) noexcept
    {
      m_data[nbh_index] = ppf;
    }

    ONIKA_HOST_DEVICE_FUNC inline const ParticlePairT<T>& get( size_t nbh_index ) const noexcept
    {
      return m_data[nbh_index];
    }

    ONIKA_HOST_DEVICE_FUNC inline const ParticlePairT<T>& operator [] ( size_t nbh_index ) noexcept
    {
      return m_data[nbh_index];
    }

    ONIKA_HOST_DEVICE_FUNC inline void copy(size_t src, size_t dst) noexcept
    {
      m_data[src] = m_data[dst];
    }
  };

  /************************************************************************************
   * Compute buffer extension for neighbor type (empty version, when type not needed
   ***********************************************************************************/
  template<typename T, size_t _MaxNeighbors>
  struct ParticleNeighborTBuffer<T, false,_MaxNeighbors>
  {
    static inline constexpr bool c_has_nbh_data = false;
    static inline constexpr size_t MaxNeighbors = _MaxNeighbors;

    ONIKA_HOST_DEVICE_FUNC static inline void set( size_t, const ParticlePairT<T>& ) noexcept {}
    ONIKA_HOST_DEVICE_FUNC static inline ParticlePairT<T> get( size_t ) noexcept { return {}; }
    ONIKA_HOST_DEVICE_FUNC inline ParticlePairT<T> operator [] ( size_t ) noexcept { return {}; }
    ONIKA_HOST_DEVICE_FUNC static inline void copy(size_t src, size_t dst) noexcept {}
  };


  /************************************************************************************
   * Neighbor type data accessor
   ***********************************************************************************/
	template<typename T>
  struct ParticleNeighborTIterator
  {
    static inline constexpr bool c_has_nbh_data = true;
    static inline constexpr ParticlePairT<T> c_default_value = {};
    CellParticleNeighborT<T> * const m_cell_types = nullptr;
    struct ParticleNeighborIteratorCtx {};
    ONIKA_HOST_DEVICE_FUNC static inline ParticleNeighborIteratorCtx make_ctx() { return {}; }
/*
    ONIKA_HOST_DEVICE_FUNC inline ParticlePairT<T>& get(size_t cell_i, size_t p_i, size_t p_nbh_index, ParticleNeighborIteratorCtx&) const noexcept
    {
      return m_cell_types[cell_i].pair_type( p_i , p_nbh_index );
    }
*/
    ONIKA_HOST_DEVICE_FUNC inline CellParticleNeighborT<T>& get(size_t cell_i, size_t p_i, size_t p_nbh_index, ParticleNeighborIteratorCtx&) const noexcept
    {
      return m_cell_types[cell_i];
    }
  };

  /************************************************************************************
   * Neighbor type null accessor
   ***********************************************************************************/
	template<typename T>
  struct ParticleNeighborTNullIterator
  {
    static inline constexpr bool c_has_nbh_data = false;
    static inline constexpr ParticlePairT<T> c_default_value = {};
    const CellParticleNeighborT<T>* m_cell_types = nullptr;
    struct ParticleNeighborIteratorCtx {};
    ONIKA_HOST_DEVICE_FUNC static inline ParticleNeighborIteratorCtx make_ctx() { return {}; }
    ONIKA_HOST_DEVICE_FUNC static inline ParticlePairT<T> get(size_t , size_t , size_t , ParticleNeighborIteratorCtx&) noexcept { return c_default_value; }
  };


  // =======================================================================================
  // ================ Utilities to move neighbor type data during MoveParticles update 
  // =======================================================================================
	template<typename T>
  struct ParticleNeighborTGridMoveBuffer;

  /* Stores type pair data series for each particle with following structure
   * type data is stored as : id, nb_pairs, pair_data[nb_pairs]
   * data range for particle p is given by m_data_range[p]
   */
	template<typename T>
  struct ParticleNeighborTCellMoveBuffer
  {
    using Word = typename CellParticleNeighborT<T>::Word;
    std::vector< std::pair<size_t,size_t> > m_data_range; // [start,end] offset in m_type_data
    std::vector< Word > m_data; // items' neighbor type data, packed contiguously

    inline void clear()
    {
      m_data_range.clear();
      m_data.clear();
    }
    inline size_t number_of_items() const
    {
      return m_data_range.size();
    }
    inline void swap( size_t a, size_t b )
    {
      std::swap( m_data_range[a] , m_data_range[b] );
    }
    
    inline void copy_incoming_particle( const ParticleNeighborTGridMoveBuffer<T>& opt_buffer, size_t cell_i, size_t p_i);
    inline uint64_t particle_id(size_t p) const
    {
      return m_data[ m_data_range[p].first ];
    }
    inline size_t particle_pair_count(size_t p) const
    {
      return m_data[ m_data_range[p].first + 1 ];
    }
    inline std::pair< const Word* , const Word* > particle_data_range(size_t p) const
    {
      return { m_data.data() + m_data_range[p].first , m_data.data() + m_data_range[p].second };
    }
    inline const ParticlePairT<T>& pair_type(size_t p, size_t i) const
    {
      const Word * __restrict__ pf_base = m_data.data() + m_data_range[p].first + 2;
      return ( (const ParticlePairT<T>*) pf_base ) [ i ];
    }
  };

	template<typename T>
  struct ParticleNeighborTGridMoveBuffer
  {
    onika::memory::CudaMMVector< CellParticleNeighborT<T> > & m_cell_type;
    ParticleNeighborTCellMoveBuffer<T> & m_otb_buffer; // buffer for elements going outside of grid
    std::vector<ParticleNeighborTCellMoveBuffer<T>> m_cell_buffer; // incoming buffers for elements entering another cell of the grid
    
    inline ParticleNeighborTCellMoveBuffer<T>* otb_buffer() { return & m_otb_buffer; }
    inline ParticleNeighborTCellMoveBuffer<T>* cell_buffer( size_t cell_i ) { return & m_cell_buffer[cell_i]; }

    inline void initialize( size_t n_cells )
    {
      m_cell_buffer.resize( n_cells );
      for(size_t i=0;i<n_cells;i++) m_cell_buffer[i].clear();
      m_otb_buffer.clear();
    }

    // must be done following the same particle re-organisation as in move_items_across_cells.h
    inline void pack_cell_particles( size_t cell_i, const std::vector<int32_t> & packed_items, bool removed_items = true )
    {
      assert( cell_i < m_cell_type.size() );
      auto & cell_type = m_cell_type[cell_i];
      
      const size_t n_incoming_items = m_cell_buffer[cell_i].number_of_items();
      const size_t total_items = removed_items ? ( packed_items.size() + n_incoming_items ) : ( cell_type.number_of_items() + n_incoming_items );

      CellParticleNeighborT<T> pf;
      pf.initialize( total_items );

      // 1. pack existing partciles, overwritting removed ones
      size_t p = 0;
      if( removed_items )
      {
        const size_t n_items = packed_items.size();
        for(p=0;p<n_items;p++)
        {
          const auto id = cell_type.particle_id( packed_items[p] );
          const size_t n_nbh_parts = cell_type.particle_number_of_pairs( packed_items[p] );
          pf.begin_nbh_write( p , id );
          for(size_t i=0;i<n_nbh_parts;i++)
          {
            pf.push_back( cell_type.pair_type( packed_items[p] , i ) );
          }
          pf.end_nbh_write( p , id );
        }
      }
      else // copy original items as is
      {
        const size_t n_items = cell_type.number_of_items();
        for(p=0;p<n_items;p++)
        {
          const auto id = cell_type.particle_id( p );
          const size_t n_nbh_parts = cell_type.particle_number_of_pairs( p );
          pf.begin_nbh_write( p , id );
          for(size_t i=0;i<n_nbh_parts;i++)
          {
            pf.push_back( cell_type.pair_type( p , i ) );
          }
          pf.end_nbh_write( p , id );
        }
      }
      
      for(size_t i=0;i<n_incoming_items;i++)
      {
        auto id = m_cell_buffer[cell_i].particle_id(i);
        size_t n_pairs = m_cell_buffer[cell_i].particle_pair_count(i);
        pf.begin_nbh_write( p , id );
        for(size_t j=0;j<n_pairs;j++) pf.push_back( m_cell_buffer[cell_i].pair_type(i,j) );
        pf.end_nbh_write( p , id );
        ++ p;
      }
      assert( p == total_items );
      cell_type = std::move( pf );
      cell_type.shrink_to_fit();
    }
         
  };

	template<typename T>
  inline void ParticleNeighborTCellMoveBuffer<T>::copy_incoming_particle( const ParticleNeighborTGridMoveBuffer<T>& opt_buffer, size_t cell_i, size_t p_i)
  {
    const CellParticleNeighborT<T> & cell_type = opt_buffer.m_cell_type[cell_i];
    const auto [ particle_data_start , particle_data_end ] = cell_type.particle_data_range(p_i);
    assert( particle_data_end >= (particle_data_start+2) ); // must contain at least id, nb_pairs
    ssize_t n_words = particle_data_end - particle_data_start;
    m_data_range.push_back( { m_data.size() , m_data.size() + n_words } );
    m_data.insert( m_data.end() , particle_data_start , particle_data_end );
  }

  // =======================================================================================
  // ======================== migrate_cell_items interoperability ======================
  // =======================================================================================
	template<typename T>
  struct NeighborTParticleMigrationHelper
  {
    using Word = typename CellParticleNeighborT<T>::Word;

    onika::memory::CudaMMVector< CellParticleNeighborT<T> > & m_cell_type;
    ParticleNeighborTCellMoveBuffer<T> & m_otb_buffer;
    
    inline std::pair<const uint8_t*,size_t> cell_particles_data(size_t cell_i) const
    {
      if( m_cell_type.empty() ) return { nullptr , 0 };
      else 
      {
        assert( cell_i < m_cell_type.size() );
        return { m_cell_type[cell_i].storage_ptr() , m_cell_type[cell_i].storage_size() };
      }
    }
    
    // returns 2 informations :
    // - pointer to start of stream location where particle # otb_i data starts
    // - size of stream portion in bytes
    inline std::pair<const uint8_t*,size_t> otb_particles_data(size_t otb_i) const
    {
      const auto [ otb_start_w , otb_end_w ] = m_otb_buffer.particle_data_range(otb_i);
      const uint8_t* otb_start = (const uint8_t*) otb_start_w;
      const uint8_t* otb_end = (const uint8_t*) otb_end_w;
      assert( otb_end >= otb_start );
      return { otb_start , otb_end - otb_start };
    }
    
		inline std::pair<const uint8_t*,size_t> otb_particle_data(size_t otb_i) const
		{
			return otb_particle_data(otb_i);
		}
    
    inline void swap_otb_particles( size_t a, size_t b )
    {
      m_otb_buffer.swap( a , b );
    }
    
    inline size_t storage_size_for_otb_range(size_t pstart, size_t pend)
    {
      if( m_cell_type.empty() ) return 0;
      
      assert( pend > pstart );
      const size_t n_items = pend - pstart;
      size_t total_size = 0;
      for(size_t p=pstart;p<pend;p++)
      {
        auto [ ptr , sz ] = otb_particle_data(p);
        total_size += sz;
      }
      // add header so that it is compatible with CellParticleNeighborT data
      total_size += ( n_items + 2 ) * sizeof(Word);
      return total_size;
    }
    
    inline size_t serialize_otb_range( void* streamv, size_t pstart, size_t pend )
    {
      assert( pstart < pend );
      const size_t n_items = pend - pstart;
      // add header so that it is compatible with CellParticleNeighborT data
      Word* streamw = (Word *) streamv;
      streamw[0] = n_items;
      streamw[1] = 0;
      uint8_t* streamb = (uint8_t*) ( streamw + n_items + 2 );
      size_t total_size = n_items + 2; // total_size count in Word units
      for(size_t p=pstart;p<pend;p++)
      {
        const size_t n_pairs = m_otb_buffer.particle_pair_count(p);
        streamw[1] += n_pairs;

        total_size += 2; // header ahead of particle's type pairs
        streamw[2+p-pstart] = total_size;
        total_size += n_pairs * CellParticleNeighborT<T>::PairTWords;
        
        auto [ ptr , sz ] = otb_particle_data(p);
        std::memcpy( streamb , ptr , sz );
        streamb += sz;
      }
      total_size *= sizeof(Word);
      assert( total_size == storage_size_for_otb_range(pstart,pend) );
      return total_size;
    }
    
    inline void clear_cell_data(size_t cell_i)
    {
      if( ! m_cell_type.empty() )
      {
        assert( cell_i < m_cell_type.size() );
        m_cell_type[cell_i].clear();
      }
    }

    struct NullParticleIdFunc { inline constexpr uint64_t operator () (size_t) { return 0; } };

    template<class CellIndexFuncT, class CellLockFuncT, class CellUnlockFuncT, class ParticleIdFuncT = NullParticleIdFunc >
    inline void append_data_stream(const void* datav, size_t data_bytes, size_t part_seq_len, CellIndexFuncT cell_index_func, CellLockFuncT cell_lock_func, CellUnlockFuncT cell_unlock_func, ParticleIdFuncT particle_id_func = {} )
    {
      static constexpr bool has_particle_id = ! std::is_same_v<ParticleIdFuncT,NullParticleIdFunc>;
    
      const Word* dataw = (const Word*) datav;
      assert( data_bytes % sizeof(Word) == 0 );
#ifndef NDEBUG
      size_t len = data_bytes / sizeof(Word);
#endif

      const size_t n_items = dataw[0];
      assert( part_seq_len == n_items );
      
#ifndef NDEBUG
      const size_t expected_total_pairs = dataw[1];
#endif
      size_t total_pairs = 0;
      for(size_t p=0;p<n_items;p++)
      {
        size_t data_offset = dataw[p+2];
        size_t n_pairs = dataw[data_offset-1];
        if constexpr ( has_particle_id )
        {
          assert( dataw[data_offset-2] == particle_id_func(p) );
        }
        total_pairs += n_pairs;
      }
      assert( total_pairs == expected_total_pairs );
      
      size_t cur_cell = std::numeric_limits<size_t>::max();
      size_t cur_cell_start_p = 0;
      
      size_t p=0;
      for(p=0;p<n_items;p++)
      {
        size_t cell_index = cell_index_func(p);
        if( cell_index != cur_cell )
        {
          if( cur_cell != std::numeric_limits<size_t>::max() )
          {
            assert( cur_cell_start_p < p );
            //lout<<"import type to cell #"<<cur_cell<<std::endl;
            m_cell_type[cur_cell].append_data_stream_range( dataw, cur_cell_start_p, p );
            cell_unlock_func( cur_cell );
          }
          cur_cell = cell_index;
          cell_lock_func( cur_cell);
          cur_cell_start_p = p;
        }
      }
      if( cur_cell != std::numeric_limits<size_t>::max() )
      {
        //lout<<"import type to cell #"<<cur_cell<<std::endl;
        m_cell_type[cur_cell].append_data_stream_range( dataw, cur_cell_start_p, p );
        cell_unlock_func( cur_cell );
      }
    }

    inline void set_dimension( const IJK& dims )
    {
      m_cell_type.clear();
      m_cell_type.resize( dims.i * dims.j * dims.k );
    }
    
  };


  // =======================================================================================
  // ================ dump reader helper for injection of type data ====================
  // =======================================================================================
	template<typename T>
  struct ParticleTReadHelper
  {
    std::vector< std::vector< std::vector<ParticlePairT<T>> > > m_out_type; // m_out_type[cell_idx][particle_idx][type_pair_idx]
    
    std::vector<CellParticleNeighborT<T>> m_in_type; // not grid cells, just chunks of type data
    std::map< uint64_t , std::pair<size_t,size_t> > m_id_map;

    inline void read_from_stream( const uint8_t* stream_start , size_t stream_size )
    {
      m_in_type.clear();
      m_id_map.clear();

      const uint8_t* stream = stream_start;
      while( (stream-stream_start) < ssize_t(stream_size) )
      {
        CellParticleNeighborT<T> cell;
        stream = cell.read_from_stream( stream );
        assert( (stream-stream_start) <= ssize_t(stream_size) );
        m_in_type.push_back( cell );
      }
      assert( (stream-stream_start) == ssize_t(stream_size) );
      
      const size_t n_cells = m_in_type.size();
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        const size_t n_items = m_in_type[cell_i].number_of_items();
        for(size_t p_i=0;p_i<n_items;p_i++)
        {
          uint64_t id = m_in_type[cell_i].particle_id( p_i );
          m_id_map[ id ] = std::pair<size_t,size_t>{ cell_i , p_i };
        }
      }
    }
        
    void initialize(size_t n_cells)
    {
      m_out_type.resize( n_cells );
      m_in_type.clear();
      m_id_map.clear();
    }
    
    inline void append_cell_particle( size_t cell_idx , size_t p_idx , uint64_t id )
    {
      if( m_id_map.empty() ) return;
      
      assert( cell_idx < m_out_type.size() );
      assert( p_idx == m_out_type[cell_idx].size() );
      auto & new_particle_type = m_out_type[cell_idx].emplace_back();
      
      auto it = m_id_map.find(id);
      if( it != m_id_map.end() )
      {
        auto [ c , p ] = it->second;
        assert( m_in_type[c].particle_id(p) == id );
        const size_t n_pairs = m_in_type[c].particle_number_of_pairs(p);
        for(size_t i=0;i<n_pairs;i++)
        {
          new_particle_type.push_back( m_in_type[c].pair_type(p,i) );
        }
      }
    }
    
    template<class ParticleIdFuncT>
    inline void finalize( GridCellParticleNeigborT<T>& nbh_type , ParticleIdFuncT particle_id )
    {
      nbh_type.m_cell_type.clear();
      const size_t n_cells = m_out_type.size();
      nbh_type.m_cell_type.resize( n_cells );
      for(size_t i=0;i<n_cells;i++)
      {
        const size_t n_items = m_out_type[i].size();
        nbh_type.m_cell_type[i].initialize( n_items );
        for(size_t p=0;p<n_items;p++)
        {
          auto id = particle_id( i , p );
          nbh_type.m_cell_type[i].begin_nbh_write( p , id );
          const size_t n_pairs = m_out_type[i][p].size();
          for(size_t j=0;j<n_pairs;j++)
          {
            nbh_type.m_cell_type[i].push_back( m_out_type[i][p][j] );
          }
          nbh_type.m_cell_type[i].end_nbh_write( p , id );
        }
        m_out_type[i].clear();
        m_out_type[i].shrink_to_fit(); // really free memory
        nbh_type.m_cell_type[i].m_data.shrink_to_fit(); // reallocate
      }
    }
  };


}

