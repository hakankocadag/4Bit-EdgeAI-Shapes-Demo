`timescale 1ns / 1ps
module mac_unit #(
 parameter IN_WIDTH = 4,
 parameter ACC_WIDTH = 16
)(
 input wire clk,
 input wire rst_n, // Asynchronous active-low reset
 input wire mac_en, // Activation trigger from FSM
 input wire acc_clr, // Synchronous clear for new inference
 input wire [IN_WIDTH-1:0] weight, // 4-bit input from Weight ROM
 input wire [IN_WIDTH-1:0] activation, // 4-bit input from Activation RAM
 output reg [ACC_WIDTH-1:0] mac_out // 16-bit safe accumulation
);
 // Internal wires for bit-width expansion and signed logic
 wire signed [IN_WIDTH-1:0] signed_w;
 wire signed [IN_WIDTH-1:0] signed_a;
 wire signed [(IN_WIDTH*2)-1:0] product; // Automatically sizes to 8 bits
 // Enforce Two's Complement signed interpretation
 assign signed_w = $signed(weight);
 assign signed_a = $signed(activation);
 // Combinational signed multiplier
 assign product = signed_w * signed_a;

 // --- Saturation Logic ---
 // Explicit sign-extension to ACC_WIDTH bits
 wire signed [ACC_WIDTH-1:0] ext_product = {{(ACC_WIDTH - (IN_WIDTH*2)){product[(IN_WIDTH*2)-1]}}, product};
 wire signed [ACC_WIDTH-1:0] sum = $signed(mac_out) + ext_product;
 
 // Overflow detection: positive+positive=negative -> pos_overflow;
 // negative+negative=positive -> neg_overflow.
 wire pos_overflow = (~mac_out[ACC_WIDTH-1]) & (~ext_product[ACC_WIDTH-1]) & sum[ACC_WIDTH-1];
 wire neg_overflow = mac_out[ACC_WIDTH-1] & ext_product[ACC_WIDTH-1] & (~sum[ACC_WIDTH-1]);

 wire signed [ACC_WIDTH-1:0] MAX_POS = {1'b0, {(ACC_WIDTH-1){1'b1}}};
 wire signed [ACC_WIDTH-1:0] MAX_NEG = {1'b1, {(ACC_WIDTH-1){1'b0}}};

 // Sequential logic for the Accumulator Register
 always @(posedge clk or negedge rst_n) begin
 if (!rst_n) begin
 mac_out <= {ACC_WIDTH{1'b0}}; // Hard reset
 end else if (acc_clr) begin
 mac_out <= {ACC_WIDTH{1'b0}}; // Soft reset for next calculation
 end else if (mac_en) begin
 // Saturation clamping
 if (pos_overflow)
 mac_out <= MAX_POS;
 else if (neg_overflow)
 mac_out <= MAX_NEG;
 else
 mac_out <= sum;
 end
 end
endmodule
