#pragma once

namespace exaDEM
{
  // not optimal
	template<typename TMPLC, typename ParticleLocation, typename FieldName>
		ONIKA_HOST_DEVICE_FUNC auto exadem_field_value(
				TMPLC* cells, 
				ParticleLocation& loc, 
				FieldName& fieldname)
		{
			return cells[loc.cell][fieldname][loc.p];
		}
}
