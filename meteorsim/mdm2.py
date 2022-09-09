import random

N_BITS = 8
def geom_50_sample():
    n = 0
    accum = -1
    while n == 0:
        n = random.getrandbits(N_BITS)
        accum += 1
    return 1 + (N_BITS - n.bit_length()) + accum * N_BITS

class MeteorEventScheduler:
    def __init__(self, num_cannons: int, recharge_ticks: int):
        self.tick = 0
        self.prev_tick = 0
        self.event_queue = []
        self.num_cannons = num_cannons
        self.cannon_ready_tick = [0] * num_cannons
        self.cannon_idx = 0
        self.next_meteor_event_tick = 0

        self.recharge_ticks = recharge_ticks

    def repr_cannon_ready(self) -> str:
        return ''.join(['R' if self.tick >= x else '.' for x in self.cannon_ready_tick])
    def cnt_cannon_ready(self) -> int:
        return [self.tick >= x for x in self.cannon_ready_tick].count(True)

    def try_fire_cannon(self) -> bool:
        """Returns whether a cannon can be fired"""
        result = self.tick >= self.cannon_ready_tick[self.cannon_idx]
        self.cannon_ready_tick[self.cannon_idx] += self.recharge_ticks
        self.cannon_idx = (self.cannon_idx + 1) % self.num_cannons
        return result

    def get_nxt_ready_cannon_idx(self) -> int:
        search_array = [self.tick >= x for x in self.cannon_ready_tick]
        return search_array.index(True) if True in search_array else -1

# @profile
def main():
    NUM_CANNONS = 16
    RECHARGE_TICKS = 150 * 60 # 2.5 min = 9000 ticks
    METEOR_EVENT_INTERVAL_LO = 1
    METEOR_EVENT_INTERVAL_HI = 30
    TICKS_PER_INTERVAL = 60 * 60
    PROB_HIT_METEOR = 0.8
    MAX_SIM_TICKS = 10000 * 60 * 60 * 60 # 1000 hours

    DPRINT = False

    event_scheduler = MeteorEventScheduler(NUM_CANNONS, RECHARGE_TICKS)
    event_idx = 0

    tot_num_meteors = 0
    tot_num_meteors_destroyed = 0
    tot_num_meteors_remain = 0
    tot_num_fires = 0
    tot_num_hits = 0
    tot_num_misses = 0
    tot_events = 0
    tot_events_success = 0
    tot_events_fail = 0

    while event_scheduler.tick <= MAX_SIM_TICKS:
        tot_events += 1
        if DPRINT: print(f'Event {event_idx}:')
        if DPRINT: print(f'    The time is: {event_scheduler.tick} ticks')

        if DPRINT: print(f'    Cannons ready: {event_scheduler.cnt_cannon_ready()} {event_scheduler.repr_cannon_ready()}')
        num_meteors = geom_50_sample()

        if DPRINT: print(f'    Incoming meteors: {num_meteors}')

        tot_num_meteors += num_meteors
        num_meteors_outstanding = num_meteors
        
        can_fire = event_scheduler.try_fire_cannon()

        while can_fire:
            if num_meteors_outstanding > 0:
                if DPRINT: print(f'    Firing cannon {nxt_cannon_idx}')
                tot_num_fires += 1
                if random.random() < PROB_HIT_METEOR:
                    num_meteors_outstanding -= 1
                    tot_num_hits += 1
                    tot_num_meteors_destroyed += 1
                    if DPRINT: print(f'        Cannon {nxt_cannon_idx} hit meteor!')
                else:
                    tot_num_misses += 1
                    if DPRINT: print(f'        Cannon {nxt_cannon_idx} missed meteor!')
                if DPRINT: print(f'        Cannon {nxt_cannon_idx} will be ready at t = {event_scheduler.cannon_ready_tick[nxt_cannon_idx]}')
            else:
                break
            can_fire = event_scheduler.try_fire_cannon()
        if DPRINT: print(f'    Remaining meteors: {num_meteors_outstanding}')
        tot_num_meteors_remain += num_meteors_outstanding
        if num_meteors_outstanding == 0:
            if DPRINT: print(f'    Success!')
            tot_events_success += 1
        else:
            if DPRINT: print(f'    Fail!')
            tot_events_fail += 1

        nxt_increment = random.randint(METEOR_EVENT_INTERVAL_LO, METEOR_EVENT_INTERVAL_HI)
        nxt_increment_ticks = nxt_increment * TICKS_PER_INTERVAL
        event_scheduler.next_meteor_event_tick = event_scheduler.next_meteor_event_tick + nxt_increment_ticks

        if DPRINT: print()
        if DPRINT: print(f'The next meteor shower is scheduled in {nxt_increment} minutes at t = {event_scheduler.next_meteor_event_tick}')
        if DPRINT: print()
        
        if DPRINT: print(f'Advancing to: {event_scheduler.next_meteor_event_tick} ticks')
        for i,v in enumerate(event_scheduler.cannon_ready_tick):
            if event_scheduler.tick < v and event_scheduler.next_meteor_event_tick >= v:
                if DPRINT: print(f'    Cannon {i} finished recharging at t = {v}')
        if DPRINT: print()
        event_scheduler.prev_tick = event_scheduler.tick
        event_scheduler.tick = event_scheduler.next_meteor_event_tick

        event_idx += 1

    print(f'The simulation terminated in {MAX_SIM_TICKS/60/60/60} hours at t = {MAX_SIM_TICKS}')
    print()
    
    print(f'Parameters:')
    print(f'    NUM_CANNONS: {NUM_CANNONS}')
    print(f'    RECHARGE_TICKS: {RECHARGE_TICKS}')
    print(f'    METEOR_EVENT_INTERVAL_LO: {METEOR_EVENT_INTERVAL_LO}')
    print(f'    METEOR_EVENT_INTERVAL_HI: {METEOR_EVENT_INTERVAL_HI}')
    print(f'    TICKS_PER_INTERVAL: {TICKS_PER_INTERVAL}')
    print(f'    PROB_HIT_METEOR: {PROB_HIT_METEOR}')
    print(f'    MAX_SIM_TICKS: {MAX_SIM_TICKS}')
    print()

    print(f'Statistics:')
    print(f'    Meteor events (total): {tot_events}')
    print(f'        Meteor events (successfully defended): {tot_events_success}')
    print(f'        Meteor events (failed to defend): {tot_events_fail}')
    print(f'        Meteor events (success%): {tot_events_success/tot_events*100}')
    print(f'    Meteors (total): {tot_num_meteors}')
    print(f'        Meteors (destroyed): {tot_num_meteors_destroyed}')
    print(f'        Meteors (remaining): {tot_num_meteors_remain}')
    print(f'        Meteors (destroy%): {tot_num_meteors_destroyed/tot_num_meteors*100}')
    print(f'    Cannon fires (total): {tot_num_fires}')
    print(f'        Cannon fires (hits): {tot_num_hits}')
    print(f'        Cannon fires (misses): {tot_num_misses}')
    print(f'        Cannon fires (hit%): {tot_num_hits/tot_num_fires*100}')

if __name__=='__main__':
    main()
