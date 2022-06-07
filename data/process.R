
get.top.taxa <- function(data,all.taxa, ntop){
    tots <- apply(data,MARGIN=2,sum)
    itots <- order(tots, decreasing=T)
    ##print(itots)
    stots <- tots[itots]

    ##pdf("plot-population-fractions.pdf")
    ##plot(cumsum(stots)/sum(tots), xlab="OTU", ylab="fraction of total")
    ##dev.off()

    #ntop <- 10
    top.taxa <- all.taxa[itots][1:ntop]
    cat( sample, ": top n :", top.taxa, "\n", sep=" ")

    M <- as.matrix( ds[,top.taxa] )
    fmiss <- length(which(M==0))/( dim(M)[1]*dim(M)[2] )
        
    cat( sample, ": missingness:", fmiss , "\n", sep=" ")

    return( list(top.taxa, fmiss) )
}

if(1){

    d <- read.csv("trimmed.csv")

    d <- d[,c(2,3,23:185)]

    dd <- split(d,d$subjectID)
    ns <- length(dd)

    fmiss.5 <- rep(0,ns)
    fmiss.8 <- rep(0,ns)
    fmiss.10 <- rep(0,ns)
    samples <- rep("",ns)
    ntime <- rep(0,ns)
    
    for(i in c(1:ns) ){
        ## write out individual level data
        sample <- as.character(dd[[i]][1,2])
        samples[i] <- sample
        fname <- paste("data-",sample,".csv",sep="")
        print(fname)

        ds <- dd[[i]]
        ds <- ds[order(ds$timepoint),]

        ntime[i] <- nrow(ds)
        
        write.table(ds, file=fname, row.names=F, col.names=T, sep=",", quote=FALSE)

        all.taxa <- names(ds)[-c(1,2)]

        ret <- get.top.taxa(ds[,-c(1,2)], all.taxa, ntop=10)
        top.taxa.10 <- ret[[1]]
        fmiss.10[i] <- ret[[2]]
        dds <- ds[,c("timepoint","subjectID", top.taxa.10)]
        write.table(dds, file=paste("data-top10-",sample,".csv",sep=""), row.names=F, col.names=T, sep=",", quote=FALSE)

        ret <- get.top.taxa(ds[,-c(1,2)], all.taxa, ntop=5)
        top.taxa.5 <- ret[[1]]
        fmiss.5[i] <- ret[[2]]
        dds <- ds[,c("timepoint","subjectID", top.taxa.5)]
        write.table(dds, file=paste("data-top5-",sample,".csv",sep=""), row.names=F, col.names=T, sep=",", quote=FALSE)

        ret <- get.top.taxa(ds[,-c(1,2)], all.taxa, ntop=8)
        top.taxa.8 <- ret[[1]]
        fmiss.8[i] <- ret[[2]]
        dds <- ds[,c("timepoint","subjectID", top.taxa.8)]
        write.table(dds, file=paste("data-top8-",sample,".csv",sep=""), row.names=F, col.names=T, sep=",", quote=FALSE)
    }

    dinfo <- data.frame(sample=samples,ntime=ntime,fmiss.10=fmiss.10, fmiss.8=fmiss.8, fmiss.5=fmiss.5)
    write.table(dinfo, "data-info.csv", row.names=F, col.names=T, sep=",", quote=FALSE)

    print( dinfo[ dinfo$fmiss.5 < 0.1,] )
}
