library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)

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
  cat( subject, ": top ", ntop, " :", top.taxa, "\n", sep=" ")

  M <- as.matrix( ds[,top.taxa] )
  fmiss <- length(which(M==0))/( dim(M)[1]*dim(M)[2] )

  cat( subject, ": missingness:", fmiss , "\n", sep=" ")

  return( list(top.taxa, fmiss) )
}

add_pseudocounts <- function(data){
  data %>%
    pivot_longer(cols = starts_with('taxa_')) %>%
    group_by(timepoint, subjectID) %>%
    mutate(value = (.data$value + 1e-3) / (1 + 1e-3 * length(unique(.data$name)))) %>%
    pivot_wider(names_from = name, values_from = value)
}

if(1){

  d <- read.csv("trimmed.csv")
  q <- read.csv('queries.csv')
  p <- read.csv('perturbations.csv') %>%
    filter(! subjectID %in% q$subjectID) %>%
    mutate(name = as.factor(name))

  # normalise reads
  d <- d %>%
    filter(! subjectID %in% q$subjectID) %>%
    pivot_longer(cols = starts_with('taxa_')) %>%
    group_by(sampleID) %>%
    mutate(value = value / sum(value)) %>%
    ungroup()

  ggplot() +
    geom_area(data = d,
             aes(timepoint, value, fill = name),
             position = 'stack') +
    geom_segment(data = p,
                 aes(x = start, xend = end, y = 1+as.numeric(name)/10, yend = 1+as.numeric(name)/10,
                     colour = name),
                 size = 1) +
    scale_fill_discrete(guide = 'none') +
    facet_wrap(~subjectID, scales = 'free_x') +
    # guides(colour = guide_legend(override.aes = list(alpha = 1))) +
    theme_bw()

  d <- d  %>%
    pivot_wider(names_from = name, values_from = value)

  d <- d[,c(2,3,23:185)]

  dd <- split(d,d$subjectID)
  ns <- length(dd)

  subjects <- rep("",ns)
  ntime <- rep(0,ns)

  top_n <- c(5, 8, 10)
  dinfo <- c()

  for(i in c(1:ns) ){
    ## write out individual level data
    subject <- as.character(dd[[i]][1,2])
    subjects[i] <- subject
    fname <- paste("data-",subject,".csv",sep="")
    print(fname)

    ds <- dd[[i]]
    ds <- ds[order(ds$timepoint),]

    ntime[i] <- nrow(ds)

    write.table(ds, file=fname, row.names=F, col.names=T, sep=",", quote=FALSE)

    all.taxa <- names(ds)[-c(1,2)]

    for(n in top_n){
      ret <- get.top.taxa(ds[,-c(1,2)], all.taxa, ntop=n)
      top.taxa <- ret[[1]]
      fmiss <- ret[[2]]

      dinfo <- rbind(dinfo,
                     data.frame(subjectID=subject,
                                ntime=nrow(ds),
                                top_n=n,
                                fmiss=fmiss))

      dds <- ds[,c("timepoint","subjectID", top.taxa)]
      dds <- add_pseudocounts(dds)

      smoothed <- dds %>%
        pivot_longer(cols = starts_with('taxa_')) %>%
        nest(timepoint, value) %>%
        mutate(m = purrr::map(data, ~ksmooth(x = .$timepoint,
                                             y = .$value,
                                             bandwidth = 3,
                                             kernel = 'normal',
                                             x.points = .$timepoint))) %>%
        select(-data)  %>%
        unnest_wider(m) %>%
        unnest_longer(c(x, y)) %>%
        rename(timepoint = x, f_value = y)

      ggplot() +
        geom_point(data = dds %>% pivot_longer(cols = starts_with('taxa_')),
                   aes(timepoint, value, colour = name)) +
        geom_line(data = smoothed,
                  aes(timepoint, f_value, colour = name)) +
        theme_bw(base_size = 8)

      ggsave(paste0("plot-top", n, "-",subject,".pdf"), height = 80, width = 80, units = 'mm')

      out <- dds %>%
        pivot_longer(cols = starts_with('taxa_')) %>%
        left_join(smoothed) %>%
        select(-value) %>%
        pivot_wider(names_from = name, values_from = f_value)

      write.table(out, file=paste0("data-top", n, "-",subject,".csv"), row.names=F, col.names=T, sep=",", quote=FALSE)
    }
  }

  write.table(dinfo, "data-info.csv", row.names=F, col.names=T, sep=",", quote=FALSE)

  # print( dinfo[ dinfo$fmiss.5 < 0.1,] )
}
