#!/usr/bin/env nextflow

params.out_dir
params.index
params.proc
params.sra_file





process download_SRA {
    conda params.conda_env_path

    input:
        val sra_id

    output:
        path "${sra_id}/${sra_id}.sra"

    script:
    """
    prefetch ${sra_id}
    """
}


process sra2fastq {
    conda params.conda_env_path

    input:
        path sra_path

    output:
        path "${sra_path.baseName}_1.fastq", emit: forward_fastq_path 
        path "${sra_path.baseName}_2.fastq", emit: reverse_fastq_path 
        val "${sra_path.baseName}"

    script:
    """
    fastq-dump --split-files $sra_path
    """
}

process bowtie2 {
    conda params.conda_env_path

    input:
        val index
        path forward_reads
        path reverse_reads
        val id
        
    output:
        path "${id}.bam", emit: aligned_reads
        
    script:
    """
    bowtie2 -p ${params.proc} -x ${index} -1 ${forward_reads} -2 ${reverse_reads} -S "${id}.bam"
    """
}

process filterBam {
    conda params.conda_env_path
    publishDir "${bam_path.baseName}", mode: 'copy'

    input:
        path bam_path
        
    output:
        path "${bam_path.baseName}.99_147_bam"
        
    script:
    """
    # Фильтрация для флага 99
    samtools view -@ ${params.proc} -h -f 99 -b ${bam_path} > tmp_99.bam
    
    # Фильтрация для флага 147
    samtools view -@ ${params.proc} -h -f 147 -b ${bam_path} > tmp_147.bam
    
    # Объединение отфильтрованных файлов
    samtools merge -@ ${params.proc} "${bam_path.baseName}.99_147_bam" tmp_99.bam tmp_147.bam
    
    # Удаление временных файлов 
    rm tmp_99.bam tmp_147.bam
    """
}

process sortBam {
    conda params.conda_env_path
    publishDir "${bam_path.baseName}", mode: 'copy'
    
     
    input:
        path bam_path
        
    output:
        path "${bam_path.baseName}.sorted_bam", emit: sorted_bam
        
    script:
    """
    samtools sort -@ 1 ${bam_path} > "${bam_path.baseName}.sorted_bam"
    """
}

process make_bam_index {
    conda params.conda_env_path
    publishDir "${bam_path.baseName}", mode: 'copy'
    
    input:
        path bam_path
        
    output:
        path "${bam_path.baseName}.bai" 
        
    script:
    """
    samtools index -@ ${params.proc} -b ${bam_path} "${bam_path.baseName}.bai" 
    """
}




workflow {

    sra_file_ch = Channel.fromPath(params.sra_file)
                      .splitCsv(header: false, sep: '\t')
                      .map { it[0] }
                        
    sra_file_ch.view()
    sra_ch = download_SRA(sra_file_ch)
    
    fastq_reads = sra2fastq(sra_ch)
    
    bam_ch = bowtie2(params.index, fastq_reads[0], fastq_reads[1], fastq_reads[2])
    
    filtered_bam_ch = filterBam(bam_ch)
    
    filtered_sorted_bam_ch = sortBam(filtered_bam_ch) 

    
    
    
    

        

}

