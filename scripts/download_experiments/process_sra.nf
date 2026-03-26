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
    fastq-dump --split-3 $sra_path
    """
}

process bowtie2 {
    conda params.conda_env_path
    publishDir "${id}", mode: 'copy'

    input:
        val index
        path forward_reads
        path reverse_reads
        val id
        
    output:
        path "${id}.bam", emit: aligned_reads
        
    script:
    """
    bowtie2 -p ${task.cpus} -x ${index} -1 ${forward_reads} -2 ${reverse_reads} -S "${id}.bam"
    """
}

process fastQC {
    conda params.conda_env_path
    publishDir "${params.out_dir}/fastqc", mode: 'copy'
    
    input:
        path bam_path
        
    output:
        path "*_fastqc.*"
        
    script:
    """
    fastqc $bam_path -t ${params.proc}
    """
}

process multiQc {
    conda params.conda_env_path
    publishDir "${params.out_dir}/multiqc", mode: 'copy'
    
    input:
        path fastqc_results
    
    output:
        path "multiqc_report.html"
        path "multiqc_data"
        
    script:
    """
    multiqc . -o .
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
    samtools view -@ ${task.cpus} -h -f 99 -b ${bam_path} > tmp_99.bam
    samtools view -@ ${task.cpus} -h -f 147 -b ${bam_path} > tmp_147.bam
    samtools merge -@ ${task.cpus} "${bam_path.baseName}.99_147_bam" tmp_99.bam tmp_147.bam
    rm tmp_99.bam tmp_147.bam
    """
}


process filterTlen {
    conda params.conda_env_path
    
    input:
        path input_bam
        
    output:
        path "${input_bam.baseName}.filtered_tlen"
        
    script:
        if (params.min_tlen != null && params.max_tlen != null) {
            """
            bamtools filter -in $input_bam -out "${input_bam.baseName}.filtered_tlen" -insertSize ">=$params.min_tlen" -insertSize "<=$params.max_tlen"
            """
            }
            
        else if (params.min_tlen == null) {
            """
            bamtools filter -in $input_bam -out "${input_bam.baseName}.filtered_tlen" -insertSize "<=$params.max_tlen" -insertSize ">=0" 
            """
            }
            
        else if (params.max_tlen == null) {
            """
            bamtools filter -in $input_bam -out "${input_bam.baseName}.filtered_tlen" -insertSize ">=$params.min_tlen"
            """
            }
        else {
            cp $input_bam "${input_bam.baseName}.filtered_tlen"
        }
            
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
    samtools sort -@ ${task.cpus} ${bam_path} > "${bam_path.baseName}.sorted_bam"
    """
}


process bam_coverage_bedgraph {
    conda params.conda_model_path
    publishDir "${bam_path.baseName}", mode: 'copy'
    
    input:
        path bam_path
        
    output:
        path "${bam_path.baseName}.bedgraph"

    script:
    """
    samtools index -@ ${task.cpus} -b ${bam_path}
    bamCoverage -b $bam_path -o "${bam_path.baseName}.bedgraph" --binSize 1 --extendReads --numberOfProcessors ${task.cpus} --outFileFormat bedgraph 
    """
}


process bam_coverage_bigwig_normed {
    conda params.conda_env_path
    publishDir "${bam_path.baseName}", mode: 'copy'
    
    input:
        path bam_path
        
    output:
        path "${bam_path.baseName}_cpm.bw"

    script:
    """
    samtools index -@ ${task.cpus} -b ${bam_path}
    bamCoverage -b $bam_path -o "${bam_path.baseName}_cpm.bw" --binSize 1 --extendReads --numberOfProcessors ${task.cpus} --normalizeUsing CPM
    """
}


process bam_coverage_bigwig {
    conda params.conda_env_path
    publishDir "${bam_path.baseName}", mode: 'copy'
    
    input:
        path bam_path
        
    output:
        path "${bam_path.baseName}.bw"

    script:
    """
    samtools index -@ ${task.cpus} -b ${bam_path}
    bamCoverage -b $bam_path -o "${bam_path.baseName}_cpm.bw" --binSize 1 --extendReads --numberOfProcessors ${task.cpus}
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
    samtools index -@ ${task.cpus} -b ${bam_path} "${bam_path.baseName}.bai" 
    """
}


process make_exo_errors {
    conda params.conda_model_path
    publishDir "${bam_path.baseName}", mode: 'copy'
    
    input:
         path bam_path
         
    output:
        path "${bam_path.baseName}_errors" 
        
    script:
    """
    python3 ${params.fitexo} -i ${bam_path} -o ./ -@ ${task.cpus} -s 0 -e 0.2 -st 0.02 -n 3
    """
}


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
    
    make_exo_errors(filtered_sorted_bam_ch)
    
    make_bam_index(filtered_sorted_bam_ch)
        

}
