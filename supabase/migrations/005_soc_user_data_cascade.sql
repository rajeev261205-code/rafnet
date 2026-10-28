-- Account deletion must not turn a user's private rows into anonymous rows.
--
-- The original schema used `on delete set null` for the owner columns. Since
-- the backend treats `user_id is null` as the anonymous principal, deleting an
-- Auth user would make their analyses/calls/threats readable by any anonymous
-- caller. Cascade instead: deleting the profile removes that user's rows.

alter table public.analysis_results
  drop constraint if exists analysis_results_user_id_fkey,
  add constraint analysis_results_user_id_fkey
    foreign key (user_id) references public.profiles(id) on delete cascade;

alter table public.calls
  drop constraint if exists calls_user_id_fkey,
  add constraint calls_user_id_fkey
    foreign key (user_id) references public.profiles(id) on delete cascade;

alter table public.threats
  drop constraint if exists threats_user_id_fkey,
  add constraint threats_user_id_fkey
    foreign key (user_id) references public.profiles(id) on delete cascade;
