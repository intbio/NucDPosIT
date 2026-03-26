process run_fimo {   
    input:
        path motif
        path genome
    
    output:
        path "${motif.baseName}.tsv"
        
    script:
    """
        fimo $motif $genome
        cp fimo_out/fimo.tsv ./
        mv fimo.tsv "${motif.baseName}.tsv"
    """
}


process convert_tsv_to_bed {
    publishDir params.out_dir, mode: 'copy' 
    
    input:
        path tsv_path
        
    output:
        path "${tsv_path.baseName}.bed"
        
    script:
        """
        python3 $params.convert_to_bed -i $tsv_path -o "${tsv_path.baseName}.bed"
        """        
}


process sort_bed {
    conda "/home/d_ryabov/.conda/envs/bedtools/"

    input:
        path bed_path
        
    output:
        path "${bed_path.baseName}.sortedbed"
        
    script:
    """
    bedtools sort  -i $bed_path > "${bed_path.baseName}.sortedbed"
    """
}



workflow {
    
    motif_ch = Channel.fromPath("${params.motif_dir}/*.meme")
    fimo_dir_ch = run_fimo(motif_ch, params.genome)
    motif_bed_ch = convert_tsv_to_bed(fimo_dir_ch)
    sorted_motif_bed_ch = sort_bed(motif_bed_ch)

}