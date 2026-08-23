# ネットワーク遮断・読み取り専用の最小環境で $1 のコマンドを実行するサンドボックス
bwrap \
--unshare-user-try \
--unshare-net \
--new-session \
--ro-bind /usr /usr \
--ro-bind /bin /bin \
--ro-bind /lib /lib \
--ro-bind /lib64 /lib64 \
--ro-bind /etc /etc \
--tmpfs /tmp \
--proc /proc \
--dev /dev \
--setenv PATH /bin:/usr/bin:/bin \
--bind /home/esakai/dev/claude/20260822/sandbox /sandbox \
--chdir /sandbox \
bash -c "$1"
