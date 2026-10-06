`timescale 1ns / 1ps
module memory_block #(
 parameter DATA_WIDTH = 4,
 parameter ADDR_WIDTH = 4,
 parameter MEM_DEPTH = 16,
 parameter INIT_FILE = "weights.hex"
)(
 input wire clk,
 input wire mem_en, // Driven by FSM
 input wire [ADDR_WIDTH-1:0] addr, // Driven by AGU
 output reg [DATA_WIDTH-1:0] read_data
);
 // 2D Array declaration for memory storage
 reg [DATA_WIDTH-1:0] mem_array [0:MEM_DEPTH-1];
 // Simulation initialization block
 // Loads the quantized neural network parameters from disk
 initial begin
 $readmemh(INIT_FILE, mem_array);
 end
 // Synchronous memory access block
 // Data is retrieved exactly one clock cycle after mem_en is asserted
 always @(posedge clk) begin
 if (mem_en) begin
 read_data <= mem_array[addr];
 end
 end
endmodule
