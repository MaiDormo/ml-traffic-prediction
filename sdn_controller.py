from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet


class SimpleSwitch(app_manager.RyuApp):
    """Basic learning switch controller"""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch, self).__init__(*args, **kwargs)
        self.mac_table = {}  # {switch_id: {mac_address: port}}
        self.logger.info("SDN Controller initialized")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def handle_switch_connection(self, ev):
        """Install default table-miss flow when switch connects"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.info(f"Switch connected: ID={datapath.id}")
        
        # Send unknown packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, 
                                         ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, priority=0, match=match, actions=actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def handle_packet_in(self, ev):
        """Process packets sent to controller"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        # Parse packet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignore LLDP (link discovery) packets
        if eth.ethertype == 0x88cc:
            return

        switch_id = datapath.id
        src_mac = eth.src
        dst_mac = eth.dst

        # Learn source MAC address
        self.mac_table.setdefault(switch_id, {})
        self.mac_table[switch_id][src_mac] = in_port

        # Determine output port
        if dst_mac in self.mac_table[switch_id]:
            out_port = self.mac_table[switch_id][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD  # Broadcast if unknown

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow to avoid future PACKET_IN messages
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, 
                                   eth_dst=dst_mac, 
                                   eth_src=src_mac)
            self._add_flow(datapath, priority=1, match=match, actions=actions,
                          idle_timeout=10, hard_timeout=30)

        # Forward the packet
        self._send_packet(datapath, msg.buffer_id, in_port, actions, msg.data)

    def _add_flow(self, datapath, priority, match, actions, 
                  idle_timeout=0, hard_timeout=0):
        """Install a flow entry in the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, 
                                                     actions)]
        flow_mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=instructions,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        datapath.send_msg(flow_mod)

    def _send_packet(self, datapath, buffer_id, in_port, actions, data):
        """Send packet out through the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        if buffer_id == ofproto.OFP_NO_BUFFER:
            data_to_send = data
        else:
            data_to_send = None

        packet_out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=buffer_id,
            in_port=in_port,
            actions=actions,
            data=data_to_send
        )
        datapath.send_msg(packet_out)
