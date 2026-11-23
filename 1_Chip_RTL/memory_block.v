`timescale 1ns / 1ps
// =============================================================================
//  Synchronous Memory Block (1-Cycle Read Latency, Optional HEX Init)
// =============================================================================
module memory_block #(
    parameter DATA_WIDTH = 4,
    parameter ADDR_WIDTH = 4,
    parameter MEM_DEPTH  = 16,
    parameter INIT_FILE  = ""
)(
    input  wire                  clk,
    input  wire                  mem_en,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg  [DATA_WIDTH-1:0] read_data
);
    reg [DATA_WIDTH-1:0] mem_array [0:MEM_DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem_array);
        end
    end

    // Synchronous read: data appears 1 clock cycle after mem_en is asserted
    always @(posedge clk) begin
        if (mem_en) begin
            read_data <= mem_array[addr];
        end
    end

endmodule
