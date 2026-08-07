include {download_SRA; sra2fastq; bowtie2; fastQC; multiQc; \
coverage_stats; filterBam; sortBam; make_exo_errors; make_bam_index}\
 from "./modules/process_sra.nf"


workflow {

    sra_file_ch = Channel.fromPath(params.sra_file)
                      .splitCsv(header: false, sep: '\t')
                      .map { it[0] }
                        
    sra_ch = download_SRA(sra_file_ch)
        
    fastq_reads = sra2fastq(sra_ch)
    
    bam_ch = bowtie2(params.index, fastq_reads[0], fastq_reads[1], fastq_reads[2])
    
    fast_qc_ch = fastQC(bam_ch)
    
    multiQc(fast_qc_ch.collect())
    
    filtered_bam_ch = filterBam(bam_ch)
    
    filtered_sorted_bam_ch = sortBam(filtered_bam_ch) 

    coverage_stats(filtered_sorted_bam_ch)
    
    make_bam_index(filtered_sorted_bam_ch)

    make_exo_errors(filtered_sorted_bam_ch)

}