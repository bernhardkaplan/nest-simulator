/*
 *  bcpnn_connection.h
 *
 *  Written by Philip Tully
 *
 */

#ifndef BCPNN_CONNECTION_H
#define BCPNN_CONNECTION_H

/* BeginDocumentation
  Name: bcpnn_synapse - Synapse type for incremental, Bayesian spike-timing 
   dependent plasticity.

  Description:
   bcpnn_synapse is a connector to create synapses with incremental, Bayesian 
   spike timing dependent plasticity.

   tau_i	double - Primary trace presynaptic time constant
   tau_j	double - Primary trace postsynaptic time constant
   tau_e	double - Secondary trace time constant
   tau_p	double - Tertiarty trace time constant
   p_i		double - \
   p_j		double -  >- these 3 initial conditions determine weight, i.e. log(p_ij/(p_i * p_j)).
   p_ij		double - /
   K_		double - Print-now signal // Neuromodulation. Turn off learning, K = 0.
   fmax_        double - Frequency assumed as maximum firing, for match with abstract rule
   epsilon_     double - lowest possible probability of spiking, e.g. lowest assumed firing rate
   bias_        double - ANN interpretation. Only calculated here to demonstrate match to rule. 
                         Will be eliminated in future versions, where bias will be calculated postsynaptically
   gain_    double - Coefficient to scale weight as conductance, can be zero-ed out
	K_values_ vector of doubles storing the recent changes of K_ since the last pre spike occured, cleared after each send function

  Transmits: SpikeEvent
   
  References:
   [1] Tully, Philip J., Matthias H. Hennig, and Anders Lansner. 
   "Synaptic and nonsynaptic plasticity approximating probabilistic inference." 
   Frontiers in synaptic neuroscience 6 (2014).
   [2] Wahlgren and Lansner (2001) Biological Evaluation of a Hebbian-Bayesian
       learning rule. Neurocomputing, 38-40, 433-438

   [3] Bergel, Transforming the BCPNN Learning Rule for Spiking Units to a
       Learning Rule for Non-Spiking Units (2010). KTH Masters Thesis.

  FirstVersion: November 2011
  CurrentVersion: August 2017
  Authors: Philip Tully, Bernhard Kaplan
          tully@csc.kth.se, bernhard.kaplan@gmail.com
  SeeAlso: synapsedict, stdp_synapse, tsodyks_synapse, static_synapse
*/

/* for Debugging */
#include <iostream>
using namespace std;

#include "connection.h"
#include "archiving_node.h"
#include "genericmodel.h"
#include <cmath>

namespace nest
{

  template < typename targetidentifierT >
  class BCPNNConnection : public nest::Connection< targetidentifierT >
  {
    public:
	  typedef CommonSynapseProperties CommonPropertiesType; // required by connector_model
	  typedef Connection< targetidentifierT > ConnectionBase;

      /* Default Constructor. Sets default values for all parameters. Needed by GenericConnectorModel. */
      BCPNNConnection();

      /* Copy constructor. Needs to be defined properly in order for GenericConnector to work. */
      BCPNNConnection(const BCPNNConnection &);

      /* Default Destructor. */
      ~BCPNNConnection() {}
      			
	  // Explicitly declare all methods inherited from the dependent base
	  // ConnectionBase. This avoids explicit name prefixes in all places these
	  // functions are used. Since ConnectionBase depends on the template parameter,
	  // they are not automatically found in the base class.
	  using ConnectionBase::get_delay_steps;
	  using ConnectionBase::get_delay;
	  using ConnectionBase::get_rport;
	  using ConnectionBase::get_target;

	  class ConnTestDummyNode : public ConnTestDummyNodeBase
	  {
	  public:
		// Ensure proper overriding of overloaded virtual functions.
		// Return values from functions are ignored.
		using ConnTestDummyNodeBase::handles_test_event;
		port
		handles_test_event( SpikeEvent&, rport )
		{
		  return invalid_port_;
		}
	  };
	  /* 
	   * This function calls check_connection on the sender and checks if the
	   * receiver accepts the event type and receptor type requested by the sender.
	   * Node::check_connection() will either confirm the receiver port by returning
	   * true or false if the connection should be ignored.
	   * We have to override the base class' implementation, since for STDP
	   * connections we have to call register_stdp_connection on the target neuron
	   * to inform the Archiver to collect spikes for this connection.
	   *
	   */
	  void check_connection(
			  Node & s, 
			  Node & t, 
			  rport receptor_type, 
			  double t_lastspike,
			  const CommonSynapseProperties &cp)
	  {
		  ConnTestDummyNode dummy_target;
		  ConnectionBase::check_connection_(dummy_target, s, t, receptor_type);
		  t.register_stdp_connection(t_lastspike - get_delay());
	  }

      /* Get all properties of this connection and put them into a dictionary. */
      void get_status(DictionaryDatum & d) const;

      /* Set properties of this connection from the values given in dictionary. */
      void set_status(const DictionaryDatum & d, nest::ConnectorModel &cm);

      /* Set properties of this connection from position p in the properties array given in dictionary. */
      void set_status(const DictionaryDatum & d, nest::index p, nest::ConnectorModel &cm);

      /* Send an event to the receiver of this connection.  */
      void send(Event& e, thread t, double t_lastspike, const CommonSynapseProperties &cp);

	  // setting the correct values for epsilon, eij, pij
      void set_initial_eps_eij_pij();

	  void
	  set_weight( double w )
	  {
		weight_ = w;
	  }

    private:
      /* data members of each connection */
      double stp_flag_;
      double yi_;
      double yj_;
      double taui_;
      double tauj_;
      double taue_;
      double taup_;
      double epsilon_;
      double K_;
      double bias_;
      double fmax_;
      double gain_;
      double zi_;
      double zj_;
      double ei_;
      double ej_;
      double eij_;
      double pi_;
      double pj_;
      double pij_;
      double t_k_;
      std::vector<double> times_k_changed;
      std::vector<double> post_spiketimes;
      std::vector<double> K_values_;
      double U_;
      double u_;
      double x_;
      double tau_rec_;
      double tau_fac_;
	  double weight_;
  }; /* of class BCPNNConnection */


  /* Send an event to the receiver of this connection.
   * \param e The event to send
   * \param p The port under which this connection is stored in the Connector.
   * \param t_lastspike Time point of last spike emitted 
  
   note: every time this method is called by an outside function, a presynaptic
       event has occured and is being transmitted to the postsynaptic side. */

  template < typename targetidentifierT >
  inline void 
  BCPNNConnection< targetidentifierT >::send(nest::Event& e, 
	nest::thread t,
	double t_lastspike, 
	const nest::CommonSynapseProperties &)
  {
    double t_spike = e.get_stamp().get_ms();  /* time stamp of current spike event */
    double resolution = nest::Time::get_resolution().get_ms();
    int spike_width = int (1. / resolution); 
    double spike_height = 1000.0 / fmax_;     /* normalizing to match this spiking rule to abstract = 1000/FMAX (Hz)*/
	Node* target = get_target( t );
    double dendritic_delay = get_delay(); /* delay from dendrite -> soma */

    /*double h = e.get_stamp().get_ms() - t_lastspike;  
    double f = std::exp(-h/tau_rec_);
    double u_decay = (tau_fac_ < 1.0e-10) ? 0.0 : std::exp(-h/tau_fac_);*/

    /* get spike history in relevant range (t1, t2] from post-synaptic neuron */
    std::deque<nest::histentry>::iterator start;
    std::deque<nest::histentry>::iterator finish;
	target->get_history(t_lastspike - dendritic_delay, t_spike - dendritic_delay,
                      &start, &finish);

    while (start != finish)  {/* loop until you get to last post spike */
        post_spiketimes.push_back(start->t_);
        start++;
    }    
 
    int number_iterations = (int)((t_spike - t_lastspike)/resolution);
    double K_vec_init = K_;
    if (K_values_.size() > 1) {
        K_vec_init = K_values_.front();
    }
    std::vector<double> K_vec (number_iterations, K_vec_init);

    if (K_values_.size() > 1) {
        std::vector<double>::iterator K_it = K_values_.end();
        std::vector<double>::iterator time_it = times_k_changed.end();
        if (times_k_changed.back() >= t_lastspike){ 
            K_it--; 
            time_it--;
            int idx_first = (int) ((t_spike - t_lastspike) / resolution);
            int idx_second;
            while (*time_it > t_lastspike){
                idx_second = (int) ((*time_it - t_lastspike)/ resolution);
                for (int i_k=idx_first-1; i_k >= idx_second; --i_k) {					
                    K_vec.at(i_k) = *K_it;
                } // for
                idx_first = idx_second;
                time_it--;
                K_it--;
            } // end of while
        }
        K_values_.clear();
        K_values_.push_back(K_);
        times_k_changed.clear();
        times_k_changed.push_back(*time_it);
    }
    
    /* Create a vector to represent the post spikes as a trace */
    std::vector<double> post_active (number_iterations, 0.);
    std::vector<double>::iterator post_it = post_spiketimes.begin(); 

    for (int timestep = 0; timestep < number_iterations; timestep++){
        /* CASE: Default. Neither Pre nor Post spike. */
        yi_ = 0.0; 
        yj_ = 0.0;

        /* CASE: Pre without (*OR WITH post) spike - synchronous events handled automatically. */
        if(timestep == 0 && t_lastspike != 0.) {
            yi_ = spike_height * spike_width;
        }

        // if you have any post spike at all
        if (post_spiketimes.size() > 0) { 
            if (post_it != post_spiketimes.end()) { 
                if (timestep == (int)((*post_it) - t_lastspike) / resolution){
                    yj_ = spike_height * spike_width;
                    post_it++;
                }
            }
        }

        /* Primary synaptic traces */
        zi_ += (yi_ - zi_ + epsilon_ ) * resolution / taui_;
        zj_ += (yj_ - zj_ + epsilon_ ) * resolution / tauj_;

        /* Secondary synaptic traces */
        ei_  += (zi_ - ei_) * resolution / taue_;
        ej_  += (zj_ - ej_) * resolution / taue_;
        eij_ += (zi_ * zj_ - eij_) * resolution / taue_;

        /* Tertiary synaptic traces */
        pi_  += K_vec.at(timestep) * (ei_ - pi_) * resolution / taup_;
        pj_  += K_vec.at(timestep) * (ej_ - pj_) * resolution / taup_;
        pij_ += K_vec.at(timestep) * (eij_ - pij_) * resolution / taup_;
    } /* of for */

    bias_ = std::log(pj_);
    
    if (stp_flag_ > 0.5){
        double_t h = e.get_stamp().get_ms() - t_lastspike;  
        double_t x_decay = std::exp(-h/tau_rec_);
        double_t u_decay = (tau_fac_ < 1.0e-10) ? 0.0 : std::exp(-h/tau_fac_);
        x_= 1. + (x_ -x_*u_ -1.)*x_decay; // Eq. 5 from reference [3]
        u_= U_+u_*(1.-U_)*u_decay; 
        weight_ = x_ * u_ * gain_ * (std::log(pij_ / (pi_ * pj_)));
    } else {
        weight_ = gain_ * (std::log(pij_ / (pi_ * pj_)));
    }

    /* Send the spike to the target */
	e.set_receiver(*target);
	e.set_weight(weight_);
	e.set_delay(get_delay_steps());
	e.set_rport(get_rport());
	e();

    post_spiketimes.clear();

    } //of BCPNNConnection::send

	  template < typename targetidentifierT >
	  BCPNNConnection< targetidentifierT >::BCPNNConnection() :
		ConnectionBase(),
		stp_flag_(0.0),
		yi_(0.0),             /* initial conditions */
		yj_(0.0),
		taui_(10.0),
		tauj_(10.0),
		taue_(100.0),
		taup_(1000.0),
		K_(1.0),
		bias_(0.0),
		fmax_(50.0),
		gain_(1.0),
		zi_(0.01), 		
		zj_(0.01),
		ei_(0.01),
		ej_(0.01),
		pi_(0.01),
		pj_(0.01),
		t_k_(0.0),
		U_(0.25),
		u_(U_),
		x_(U_),
		tau_rec_(600.0),
		tau_fac_(0.0),
		weight_(1.0)
	  { 
		times_k_changed.push_back(t_k_);
		K_values_.push_back(K_);
		set_initial_eps_eij_pij();
	  }

  template < typename targetidentifierT >
  BCPNNConnection< targetidentifierT >::BCPNNConnection(const BCPNNConnection &rhs) :
	ConnectionBase(),
    stp_flag_(rhs.stp_flag_),
    yi_(rhs.yi_),
    yj_(rhs.yj_),
    taui_(rhs.taui_),
    tauj_(rhs.tauj_),
    taue_(rhs.taue_),
    taup_(rhs.taup_),
    K_(rhs.K_),
    bias_(rhs.bias_),
    fmax_(rhs.fmax_),
    gain_(rhs.gain_),
    zi_(rhs.zi_),
    zj_(rhs.zj_),
    ei_(rhs.ei_),
    ej_(rhs.ej_),
    pi_(rhs.pi_),
    pj_(rhs.pj_),
    t_k_(rhs.t_k_),
    U_(rhs.U_),
    u_(rhs.u_),
    x_(rhs.x_),
    tau_rec_(rhs.tau_rec_),
    tau_fac_(rhs.tau_fac_),
	weight_(rhs.weight_)
  {
    times_k_changed.push_back(rhs.t_k_);
    K_values_.push_back(rhs.K_);
    set_initial_eps_eij_pij(); 
  }

  template < typename targetidentifierT >
  void BCPNNConnection< targetidentifierT >::set_initial_eps_eij_pij(){
    epsilon_ = 1. / (fmax_ * taup_);
    eij_ = ei_ * ej_;
    pij_ = pi_ * pj_;
  }

  template < typename targetidentifierT >
  void BCPNNConnection< targetidentifierT >::get_status(DictionaryDatum & d) const
  {
	  ConnectionBase::get_status(d);
    def<double>(d, "stp_flag", stp_flag_);
    def<double>(d, "tau_i", taui_);
    def<double>(d, "tau_j", tauj_);
    def<double>(d, "tau_e", taue_);
    def<double>(d, "tau_p", taup_);
    def<double>(d, "epsilon", epsilon_);
    def<double>(d, "fmax", fmax_);
    def<double>(d, "bias", bias_);
    def<double>(d, "K", K_);
    def<double>(d, "gain", gain_);
    def<double>(d, "p_i", pi_);
    def<double>(d, "p_j", pj_);
    def<double>(d, "p_ij", pij_);
    def<double>(d, "t_k_", t_k_);
    def<double>(d, nest::names::dU, U_);
    def<double>(d, nest::names::u, u_);
    def<double>(d, nest::names::tau_rec, tau_rec_);
    def<double>(d, nest::names::tau_fac, tau_fac_);
    def<double>(d, nest::names::x, x_);
    def<double>(d, nest::names::weight, weight_);
  }

  template < typename targetidentifierT >
  void BCPNNConnection< targetidentifierT >::set_status(const DictionaryDatum & d, nest::ConnectorModel &cm)
  {
	  ConnectionBase::set_status(d, cm);
    updateValue<double>(d, "stp_flag", stp_flag_);
    updateValue<double>(d, "tau_i", taui_);
    updateValue<double>(d, "tau_j", tauj_);
    updateValue<double>(d, "tau_e", taue_);
    updateValue<double>(d, "tau_p", taup_);
    updateValue<double>(d, "K", K_);
    updateValue<double>(d, "epsilon", epsilon_);
    updateValue<double>(d, "fmax", fmax_);
    updateValue<double>(d, "bias", bias_);
    updateValue<double>(d, "gain", gain_);
    updateValue<double>(d, "p_i", pi_);
    updateValue<double>(d, "p_j", pj_);
    updateValue<double>(d, "p_ij", pij_);
    updateValue<double>(d, "t_k", t_k_);
    updateValue<double>(d, nest::names::dU, U_);
    updateValue<double>(d, nest::names::u, u_);
    updateValue<double>(d, nest::names::tau_rec, tau_rec_);
    updateValue<double>(d, nest::names::tau_fac, tau_fac_);
    updateValue<double>(d, nest::names::x, x_);
	updateValue<double>(d, nest::names::weight, weight_);
	// only update K_values_ if K is not the same it has been initialized to
	if ((t_k_) == times_k_changed.back()){
		K_values_.pop_back();
		K_values_.push_back(K_);
	} else { 
		times_k_changed.push_back(t_k_);
		K_values_.push_back(K_);
	}
  }

   /**
   * Set properties of this connection from position p in the properties
   * array given in dictionary.
   */
  template < typename targetidentifierT >
  void BCPNNConnection< targetidentifierT >::set_status(const DictionaryDatum & d, nest::index p, nest::ConnectorModel &cm)
  {
	Connection< targetidentifierT >::set_status(d, p, cm);
//    updateValue<double>(d, names::stp_flag, p, stp_flag_);
	updateValue<double>(d, "stp_flag", stp_flag_);
//    set_property<double>(d, "stp_flag", stp_flag_);
    updateValue<double>(d, "tau_i", taui_);
    updateValue<double>(d, "tau_j", tauj_);
    updateValue<double>(d, "tau_e", taue_);
    updateValue<double>(d, "tau_p", taup_);
    updateValue<double>(d, "K", K_);
    updateValue<double>(d, "epsilon", epsilon_);
    updateValue<double>(d, "fmax", fmax_);
    updateValue<double>(d, "bias", bias_);
    updateValue<double>(d, "gain", gain_);
    updateValue<double>(d, "p_i", pi_);
    updateValue<double>(d, "p_j", pj_);
    updateValue<double>(d, "p_ij", pij_);
    updateValue<double>(d, "t_k", t_k_);
    updateValue<double>(d, nest::names::U, U_);
    updateValue<double>(d, nest::names::u, u_);
    updateValue<double>(d, nest::names::x, x_);
    updateValue<double>(d, nest::names::tau_rec, tau_rec_);
    updateValue<double>(d, nest::names::tau_fac, tau_fac_);
	if ((t_k_) == times_k_changed.back()){
		K_values_.pop_back();
		K_values_.push_back(K_);
	} else { 
		times_k_changed.push_back(t_k_);
		K_values_.push_back(K_);
	}
  }

} //of namespace nest
#endif // of #ifndef BCPNN_CONNECTION_H

