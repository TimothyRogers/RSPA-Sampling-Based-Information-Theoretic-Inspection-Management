function [] = printout(fname)


print(strcat(fname,'.eps'),'-depsc2','-r0')
print(strcat(fname,'.png'),'-dpng','-r0')

end