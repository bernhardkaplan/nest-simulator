/*
 *  bcpnn_connection.cpp
 *
 *  Written by Philip Tully, Bernhard Kaplan
 *
 */

#include "dictdatum.h"
#include "connector_model.h"
#include "common_synapse_properties.h"
#include "bcpnn_connection.h"
#include "event.h"
#include <vector>

namespace nest
{

  template < typename targetidentifierT >
  void BCPNNConnection< targetidentifierT >::initialize_property_arrays(DictionaryDatum & d) const
  {
	Connection< targetidentifierT >::initialize_property_arrays(d);
    initialize_property_array(d, "stp_flag");
    initialize_property_array(d, "tau_i");
    initialize_property_array(d, "tau_j");
    initialize_property_array(d, "tau_e");
    initialize_property_array(d, "tau_p");
    initialize_property_array(d, "K");
    initialize_property_array(d, "epsilon");
    initialize_property_array(d, "fmax");
    initialize_property_array(d, "bias");
    initialize_property_array(d, "gain");
    initialize_property_array(d, "p_i");
    initialize_property_array(d, "p_j");
    initialize_property_array(d, "p_ij");
    initialize_property_array(d, "t_k");
    initialize_property_array(d, nest::names::U);    
    initialize_property_array(d, nest::names::u); 
    initialize_property_array(d, nest::names::tau_rec);  
    initialize_property_array(d, nest::names::tau_fac);  
    initialize_property_array(d, nest::names::x);
  }

  /**
   * Append properties of this connection to the given dictionary. If the
   * dictionary is empty, new arrays are created first.
   */
  template < typename targetidentifierT >
  void BCPNNConnection< targetidentifierT >::append_properties(DictionaryDatum & d) const
  {
	Connection< targetidentifierT >::append_properties(d);
    append_property<double>(d, "stp_flag", stp_flag_);
    append_property<double>(d, "tau_i", taui_);
    append_property<double>(d, "tau_j", tauj_);
    append_property<double>(d, "tau_e", taue_);
    append_property<double>(d, "tau_p", taup_);
    append_property<double>(d, "K", K_);
    append_property<double>(d, "epsilon", epsilon_);
    append_property<double>(d, "fmax", fmax_);
    append_property<double>(d, "bias", bias_);
    append_property<double>(d, "gain", gain_);
    append_property<double>(d, "p_i", pi_);
    append_property<double>(d, "p_j", pj_);
    append_property<double>(d, "p_ij", pij_);
    append_property<double>(d, "t_k", t_k_);
    append_property<double>(d, nest::names::U, U_); 
    append_property<double>(d, nest::names::u, u_); 
    append_property<double>(d, nest::names::tau_rec, tau_rec_);  
    append_property<double>(d, nest::names::tau_fac, tau_fac_);  
    append_property<double>(d, nest::names::x, x_);
  }
} // of namespace nest
